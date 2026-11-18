"""Spatial gene embedding: a reference-free MetaSpot representation.

Genes are embedded by how often their transcripts sit next to each other
inside a cell, relative to a within-cell shuffle (positive PMI of
different-gene neighbour pairs, factorised by SVD). Each transcript is then
described by the mean embedding of its neighbours (itself excluded), centred
within its cell, and each MetaSpot by the mean over its transcripts.

Coordinates are 3D when the table has ``z_location``, otherwise 2D.

On Atera BRCA (5,000 cells) MetaDomains built on this representation separate
held-out ER, signal-peptide and nuclear-speckle transcripts better than the
DGI/legacy MetaDomains, which perform close to a geometry-only baseline.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.utils.extmath import randomized_svd

GENE_EMBEDDING_REP = "X_gene_embedding"


class TooFewGenesError(ValueError):
    """Fewer than two genes reach min_gene_count, so no embedding can be fitted."""
# Default radii as multiples of the median x-y nearest-neighbour distance, so they
# follow the data's coordinate unit and density. On Atera BRCA (µm, median
# nearest-neighbour 0.07 µm) they give ~0.7 and ~1.0 µm. Against 15x/23x (~1.0/1.6 µm)
# they raised held-out MetaDomain AUROC on 1k and, independently, 5k cells
# (5k, 3D: ER 0.554 -> 0.569, Speckle 0.534 -> 0.547).
PAIR_RADIUS_NN = 10.0
NEIGHBOUR_RADIUS_NN = 14.0
# gene_embedding_scale="multiscale" concatenates embeddings at MULTISCALE_NN (multiples of the nearest-neighbour
# distance); the neighbourhood radius keeps the 14:10 ratio.
MULTISCALE_NN = (3.0, 7.0, 14.0)


def _cell_groups(cells: np.ndarray):
    order = np.argsort(cells, kind="stable")
    return np.split(order, np.flatnonzero(cells[order][1:] != cells[order][:-1]) + 1)


def _per_cell(function, groups, arrays, n_jobs: int, **kwargs) -> list:
    """``function`` on each cell's slice of ``arrays``, in cell order; cells are independent."""
    tasks = (delayed(function)(*(a[idx] for a in arrays), **kwargs) for idx in groups)
    return Parallel(n_jobs=max(1, int(n_jobs)), batch_size=64)(tasks)


def _cell_pair_stats(xy, local, *, column, G, pair_radius):
    pairs = cKDTree(xy).query_pairs(pair_radius, output_type="ndarray")
    if not len(pairs):
        return None
    n = len(local)
    _, inverse, per_gene = np.unique(local, return_inverse=True, return_counts=True)
    same_gene_n = per_gene[inverse]
    i = np.r_[pairs[:, 0], pairs[:, 1]]
    j = np.r_[pairs[:, 1], pairs[:, 0]]
    different = local[i] != local[j]
    a, b = column[local[i[different]]], column[local[j[different]]]
    ok = (a >= 0) & (b >= 0)
    degree = np.bincount(i[different], minlength=n)
    s = column[local]
    has = s >= 0
    n_s = np.bincount(s[has], minlength=G).astype(np.float32)
    d_s = np.bincount(s[has], weights=degree[has], minlength=G)
    n_same = np.zeros(G)
    n_same[s[has]] = same_gene_n[has]
    # Expected different-gene neighbours g->h under a within-cell shuffle: W_g * n_h.
    weight = np.where(n_s > 0, d_s / np.maximum(n - n_same, 1), 0).astype(np.float32)
    # 每个细胞只返回出现过的基因对及其次数：几千个细胞的逐对列表放在一起会占满内存
    pair_codes, pair_counts = np.unique(a[ok] * G + b[ok], return_counts=True)
    return pair_codes, pair_counts, weight, n_s


def _pair_matrices(groups, xy, gene_codes, column, G, pair_radius, n_jobs):
    """Observed and within-cell-shuffle expected different-gene neighbour counts (G x G, symmetric, zero diagonal)."""
    # 边到边累加，不把所有细胞的基因对列表同时留在内存里；基因对计数是整数，累加顺序不影响结果
    tasks = (delayed(_cell_pair_stats)(xy[idx], gene_codes[idx], column=column, G=G, pair_radius=pair_radius) for idx in groups)
    results = Parallel(n_jobs=max(1, int(n_jobs)), batch_size=64, return_as="generator")(tasks)
    counts = np.zeros(G * G, dtype=np.float64)
    weights, totals = [], []
    for r in results:
        if r is None:
            continue
        counts[r[0]] += r[1]  # r[0] 在一个细胞内没有重复，所以与 np.add.at 结果相同，但快得多
        weights.append(r[2])
        totals.append(r[3])
    if not weights:
        raise ValueError("spatial gene embedding found no transcript pairs within pair_radius")
    observed = counts.reshape(G, G).astype(np.float32)
    del counts
    expected = (np.asarray(weights).T @ np.asarray(totals)).astype(np.float32)
    observed = (observed + observed.T) / 2
    expected = (expected + expected.T) / 2
    np.fill_diagonal(observed, 0)
    np.fill_diagonal(expected, 0)
    return observed, expected


def fit_gene_embedding(
    cells: np.ndarray,
    gene_codes: np.ndarray,
    xy: np.ndarray,
    n_genes: int,
    *,
    pair_radius: float = 1.0,
    dim: int = 32,
    min_gene_count: int = 100,
    pseudo_count: float = 5.0,
    max_cells: int = 20000,
    seed: int = 0,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gene index per kept gene, embedding rows) from within-cell co-localisation.

    ``gene_codes`` are integers in ``[0, n_genes)``. Only cells sampled up to
    ``max_cells`` are used to fit; genes with fewer than ``min_gene_count``
    transcripts in them get no embedding.
    """
    groups = [g for g in _cell_groups(cells) if len(g) >= 2]
    if len(groups) > max_cells:
        pick = np.random.default_rng(seed).choice(len(groups), max_cells, replace=False)
        groups = [groups[i] for i in np.sort(pick)]
    used = np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
    kept = np.flatnonzero(np.bincount(gene_codes[used], minlength=n_genes) >= min_gene_count)
    G = len(kept)
    if G < 2:
        raise TooFewGenesError(f"spatial gene embedding needs >= 2 genes with {min_gene_count} transcripts, found {G}")
    column = np.full(n_genes, -1, dtype=np.int64)
    column[kept] = np.arange(G)
    observed, expected = _pair_matrices(groups, xy, gene_codes, column, G, pair_radius, n_jobs)
    # The pseudo-count shrinks rarely observed pairs towards no association.
    ppmi = np.maximum(np.log((observed + pseudo_count) / (expected + pseudo_count)), 0).astype(np.float32)
    del observed, expected
    dim = max(1, min(int(dim), G - 1))
    u, s_values, _ = randomized_svd(ppmi, dim, random_state=seed)
    return kept, (u * np.sqrt(s_values)).astype(np.float32)


def _cell_spot_sums_many(xy, part, spot, *codes, lookups, has_vectors, neighbour_radius):
    """一个细胞的若干变体（每个变体：该细胞每行的基因编号 + 基因向量表 + 可用基因）的 MetaSpot 邻域向量和
    （float64，按 MetaSpot 编号排序）。邻居对与邻接结构只依赖位置，所有变体共用；每个变体只换邻接矩阵的取值
    （不可用的邻居记 0），矩阵乘按同样顺序累加，结果与逐个变体单独计算逐位相同。"""
    n = len(spot)
    rows = np.flatnonzero(spot >= 0)
    spots, column = np.unique(spot[rows], return_inverse=True)
    members = sparse.csr_matrix((np.ones(rows.size), (column, rows)), shape=(len(spots), n))
    pairs = cKDTree(xy).query_pairs(neighbour_radius, output_type="ndarray")
    base = None
    if len(pairs):
        i = np.r_[pairs[:, 0], pairs[:, 1]]
        j = np.r_[pairs[:, 1], pairs[:, 0]]
        base = sparse.csr_matrix((np.ones(len(i), dtype=np.float32), (i, j)), shape=(n, n))  # 行内按列排序
    sums = []
    for local, lookup, has_vector in zip(codes, lookups, has_vectors):
        out = np.zeros((n, lookup.shape[1]), dtype=np.float32)
        if base is not None:
            adjacency = sparse.csr_matrix((has_vector[local[base.indices]].astype(np.float32), base.indices, base.indptr), shape=(n, n))
            degree = np.asarray(adjacency.sum(axis=1)).ravel()
            vectors = (adjacency @ lookup[local]) / np.maximum(degree, 1)[:, None]
            for value in np.unique(part):
                sel = part == value
                out[sel] = vectors[sel] - vectors[sel].mean(axis=0)
        sums.append(members @ out.astype(np.float64))
    return spots, sums


def _metaspot_vectors_many(cells, gene_codes_list, xy, part, codes, kept_list, embedding_list, n_genes, n_spots, neighbour_radius, n_jobs):
    """_metaspot_vectors 的多变体版：每个细胞只找一次邻居对，在同一个任务里算完所有变体。
    返回每个变体一个 n_spots x dim 的 float32 数组（顺序同 gene_codes_list）。"""
    lookups, has_vectors = [], []
    for kept, embedding in zip(kept_list, embedding_list):
        lookup = np.zeros((n_genes, embedding.shape[1]), dtype=np.float32)
        lookup[kept] = embedding
        has_vector = np.zeros(n_genes, dtype=bool)
        has_vector[kept] = True
        lookups.append(lookup)
        has_vectors.append(has_vector)
    sums = [np.zeros((n_spots, lk.shape[1]), dtype=np.float64) for lk in lookups]
    for spots, cell_sums in _per_cell(
        _cell_spot_sums_many, _cell_groups(cells), (xy, part, codes, *gene_codes_list), n_jobs,
        lookups=lookups, has_vectors=has_vectors, neighbour_radius=neighbour_radius,
    ):
        for total, cell_sum in zip(sums, cell_sums):
            total[spots] += cell_sum
    n = np.maximum(np.bincount(codes[codes >= 0], minlength=n_spots), 1)[:, None]
    return [(total / n).astype(np.float32) for total in sums]


def _metaspot_vectors(cells, gene_codes, xy, part, codes, kept, embedding, n_genes, n_spots, neighbour_radius, n_jobs):
    """MetaSpot means of the transcripts' neighbourhood vectors."""
    return _metaspot_vectors_many(cells, [gene_codes], xy, part, codes, [kept], [embedding], n_genes, n_spots, neighbour_radius, n_jobs)[0]


def median_nearest_neighbour_distance(cells: np.ndarray, xy: np.ndarray, max_cells: int = 200, seed: int = 0) -> float:
    """Median distance from a transcript to its nearest other transcript in the same cell."""
    groups = [g for g in _cell_groups(cells) if len(g) >= 2]
    if not groups:
        raise ValueError("no cell has two transcripts; cannot set a neighbour radius")
    if len(groups) > max_cells:
        groups = [groups[i] for i in np.random.default_rng(seed).choice(len(groups), max_cells, replace=False)]
    dist = np.concatenate([cKDTree(xy[g]).query(xy[g], k=2)[0][:, 1] for g in groups])
    dist = dist[dist > 0]
    if not len(dist):
        raise ValueError("all transcripts share coordinates; cannot set a neighbour radius")
    return float(np.median(dist))


def resolve_radius(value, nearest: float, multiple: float) -> float:
    """``None``/"auto" -> multiple x median nearest-neighbour distance; a number is used as is (data units)."""
    if value is None or str(value).strip().lower() == "auto":
        return multiple * nearest
    return float(value)


def metaspot_gene_embedding(
    point_df: pd.DataFrame,
    metaspot_names: pd.Index,
    *,
    metaspot_col: str = "metaspot",
    pair_radius=None,
    neighbour_radius=None,
    scale: str = "density",
    dim: int = 32,
    min_gene_count: int = 100,
    pseudo_count: float = 5.0,
    max_cells: int = 20000,
    seed: int = 0,
    log=None,
    n_jobs: int = 1,
    gene_vectors: dict | None = None,
) -> np.ndarray:
    """MetaSpot x dim matrix aligned to ``metaspot_names`` (MetaSpots without transcripts are zero).

    ``gene_vectors``, when a dict, receives what a later read-out needs: ``genes``, ``vectors`` (genes x all scales),
    ``scale_dims`` and ``neighbour_radii`` (one per scale).

    Radii are in coordinate units; ``None`` or "auto" sets them from the data with ``scale``:
    "density" = fixed multiples of the median nearest-neighbour distance; "multiscale" = embeddings at
    ``MULTISCALE_NN`` concatenated (dim per scale), each block scaled to unit mean row norm.
    """
    scale = str(scale or "density").strip().lower()
    if scale not in {"density", "multiscale"}:
        raise ValueError("gene_embedding_scale must be 'density' or 'multiscale'")
    cells = pd.factorize(point_df["cell_id"].astype(str))[0]
    gene_codes, genes = pd.factorize(point_df["feature_name"].astype(str))
    # z, when the platform reports it, keeps transcripts above/below each other from counting as neighbours.
    axes = ["x_location", "y_location"]
    if "z_location" in point_df.columns and point_df["z_location"].notna().all():
        axes.append("z_location")
    xy = point_df[axes].to_numpy(np.float64)
    # The length scale comes from the x-y density in 2D and 3D alike: on Atera BRCA the
    # same physical radius worked best with z, a larger 3D radius matching the 2D
    # neighbour count was worse.
    nearest = median_nearest_neighbour_distance(cells, xy[:, :2], seed=seed)
    multiples = [(PAIR_RADIUS_NN, NEIGHBOUR_RADIUS_NN)]
    if scale == "multiscale":
        multiples = [(m, m * NEIGHBOUR_RADIUS_NN / PAIR_RADIUS_NN) for m in MULTISCALE_NN]
    compartments = point_df["in_nucleus"].to_numpy(bool) if "in_nucleus" in point_df.columns else None
    part = np.zeros(len(cells), dtype=bool) if compartments is None else compartments
    spot = pd.Categorical(point_df[metaspot_col].astype(str), categories=pd.Index(metaspot_names).astype(str))
    codes = np.asarray(spot.codes, dtype=np.int64)
    blocks, vectors, radii = [], [], []
    for pair_nn, neighbour_nn in multiples:
        pr = resolve_radius(pair_radius, nearest, pair_nn)
        nr = resolve_radius(neighbour_radius, nearest, neighbour_nn)
        if log is not None:
            log(f"Spatial gene embedding radii ({len(axes)}D, coordinate units): median nearest neighbour={nearest:.4g}, "
                f"pair={pr:.4g}, neighbourhood={nr:.4g}")
        kept, embedding = fit_gene_embedding(
            cells, gene_codes, xy, len(genes), pair_radius=pr, dim=dim,
            min_gene_count=min_gene_count, pseudo_count=pseudo_count, max_cells=max_cells, seed=seed, n_jobs=n_jobs,
        )
        # MetaSpot sums are formed inside each cell's task (a MetaSpot lies in one cell), so
        # the transcripts x dim vector table is never built: on Xenium COAD (272M
        # transcripts) it was ~70 GB and its single-threaded aggregation took >20 min.
        block = _metaspot_vectors(cells, gene_codes, xy, part, codes, kept, embedding, len(genes),
                                  len(metaspot_names), nr, n_jobs)
        if len(multiples) > 1:  # equal weight per scale
            scale_factor = max(float(np.linalg.norm(block, axis=1).mean()), 1e-12)
            block /= scale_factor
            # Keep the gene vectors in the same units as the normalized
            # MetaSpot block used by the downstream structure-axis readout.
            embedding = embedding / scale_factor
        blocks.append(block)
        vectors.append(embedding)
        radii.append(nr)
    if gene_vectors is not None:
        gene_vectors.update(genes=np.asarray(genes[kept]).astype(str), vectors=np.hstack(vectors).astype(np.float32),
                            scale_dims=np.asarray([v.shape[1] for v in vectors]), neighbour_radii=np.asarray(radii, np.float64))
    return np.hstack(blocks).astype(np.float32)
