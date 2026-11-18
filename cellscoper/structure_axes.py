"""Structure axes: a continuous score per MetaSpot and structure, read directly from the representations.

Reference marker genes only name a direction: the mean vector of a structure's markers minus the mean vector of all
genes, in the spatial gene embedding (one direction per scale) and in the decoder W of the masked-transcript model. A
MetaSpot's score is its representation projected on that direction, standardised and averaged over its nearest
MetaSpots of the same cell side. No MetaDomain is involved.

The gene-shuffled null (below) also sets how each read-out counts, without any label: gene-embedding directions are
whitened by the null's covariance (matched filter), and scales and read-outs are added with weight sqrt(1 - rho) / rho,
rho = the null's share of the read-out's variance (high at the smallest scale, lower at larger ones).

Labels: gene identities are shuffled within each cell side and the same score is recomputed, which keeps positions,
MetaSpots and every cell's RNA mixture. A structure takes the MetaSpots above the lowest score at which the null's share
above that score is at most q times the real share (the estimated share of false labels). A structure that never
reaches this is not detected and takes no MetaSpot; there are no per-cell seats. Held-out marker enrichment is reported
for every structure and does not change the labels.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .lens import smooth_neighbours_of, smooth_with
from .spatial_gene_embedding import _metaspot_vectors_many

# Scores are averaged over about this many transcripts (MetaSpots of ~10 give 5 neighbours on Atera).
AXIS_SMOOTH_TRANSCRIPTS = 50
# Ridge of the null covariance, as a share of its mean eigenvalue (results are insensitive between .01 and 1).
WHITEN_RIDGE = 0.1
# A structure needs at least this share of its side's MetaSpots above the threshold to count as detected.
MIN_LABEL_SHARE = 0.002
_Q_GRID = np.geomspace(0.5, MIN_LABEL_SHARE, 200)  # tail shares at which the false-label estimate is evaluated


def direction(vectors: np.ndarray, genes: np.ndarray, markers: Sequence[str]) -> Tuple[Optional[np.ndarray], int]:
    """Mean vector of the markers minus the mean vector of all genes, and the number of markers found."""
    use = np.isin(genes, list(markers))
    if not use.any():
        return None, 0
    return vectors[use].mean(axis=0) - vectors.mean(axis=0), int(use.sum())


def false_share_curve(real: np.ndarray, null: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(score grid, estimated false-label share of everything above each score), monotone as a q-value."""
    real, null = np.sort(real), np.sort(null)
    grid = np.quantile(real, 1 - _Q_GRID)
    above_real = 1 - np.searchsorted(real, grid, side="right") / max(real.size, 1)
    above_null = 1 - np.searchsorted(null, grid, side="right") / max(null.size, 1)
    ratio = above_null / np.maximum(above_real, 1e-12)
    return grid, np.minimum(np.minimum.accumulate(ratio), 1.0)


def threshold_for(real: np.ndarray, null: np.ndarray, q: float) -> Optional[float]:
    """Lowest score whose estimated false-label share is at most q; None when no score reaches it."""
    grid, curve = false_share_curve(real, null)
    hit = np.flatnonzero(curve <= q)
    return float(grid[hit[0]]) if hit.size else None


def smooth_neighbours(median_transcripts: float, transcripts: float = AXIS_SMOOTH_TRANSCRIPTS) -> int:
    return max(1, int(round(float(transcripts) / max(float(median_transcripts), 1.0))))


def read_weight(real: np.ndarray, null: np.ndarray) -> float:
    """Weight of a standardised read-out when noise is independent between read-outs: its signal over its noise variance."""
    real_var = float(np.var(real))
    null_var = float(np.var(null))
    if not np.isfinite(real_var) or not np.isfinite(null_var) or real_var <= 1e-12:
        return 0.0
    # Keep the weight finite for a degenerate null readout while preserving the
    # high-weight limit of a readout whose shuffled scores have almost no spread.
    rho = np.clip(null_var / real_var, 1e-6, 0.999)
    return float(np.sqrt(1 - rho) / rho)


def whitener(null_block: np.ndarray, *, max_rows: int = 50000, seed: int = 0):
    """Solver of (null covariance + ridge) d = direction."""
    rows = null_block if len(null_block) <= max_rows else null_block[np.random.default_rng(seed).choice(len(null_block), max_rows, replace=False)]
    cov = np.cov(rows.T)
    cov = np.atleast_2d(cov) + WHITEN_RIDGE * np.trace(np.atleast_2d(cov)) / len(np.atleast_2d(cov)) * np.eye(len(np.atleast_2d(cov)))
    return lambda d: np.linalg.solve(cov, d)


def text_codes(col: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """与 pd.factorize(col.astype(str)) 完全相同的 (每行编号, 编号对应文字)。categorical 列只转换类别，
    不为几亿行逐行生成字符串（缺失值与 astype(str) 一样记作 "nan"；转成相同文字的类别合并）。"""
    if not isinstance(col.dtype, pd.CategoricalDtype):
        codes, uniques = pd.factorize(col.astype(str).to_numpy())
        return codes, np.asarray(uniques).astype(str)
    text = np.append(col.cat.categories.astype(str).to_numpy(dtype=object), "nan")
    text_id, text_uniques = pd.factorize(text)
    codes, first = pd.factorize(text_id[col.cat.codes.to_numpy()])  # 编号 -1（缺失）取到最后一个 "nan"
    return codes, np.asarray(text_uniques[first]).astype(str)


class AxisData:
    """Transcript and MetaSpot tables of one run; every MetaSpot array follows ``spot_names``."""

    def __init__(self, transcripts: pd.DataFrame, spot_names: Sequence[str], *, seed: int = 0, n_jobs: int = 1, smooth_transcripts: float = AXIS_SMOOTH_TRANSCRIPTS):
        self.names = np.asarray(spot_names).astype(str)
        spot_codes, spot_text = text_codes(transcripts["metaspot"])
        self.codes = pd.Index(self.names).get_indexer(spot_text)[spot_codes].astype(np.int64)
        self.S = len(self.names)
        self.size = np.bincount(self.codes[self.codes >= 0], minlength=self.S)
        self.tx_cell = text_codes(transcripts["cell_id"])[0]
        self.gene_codes, self.gene_names = text_codes(transcripts["feature_name"])
        axes = ["x_location", "y_location"]
        if "z_location" in transcripts.columns and transcripts["z_location"].notna().all():
            axes.append("z_location")
        self.xyz = transcripts[axes].to_numpy(np.float64)
        self.part = transcripts["in_nucleus"].to_numpy(bool)
        n = np.maximum(self.size, 1)
        self.side = np.bincount(self.codes, weights=self.part.astype(float), minlength=self.S) / n > 0.5
        self.centroid = np.column_stack([np.bincount(self.codes, weights=self.xyz[:, j], minlength=self.S) / n for j in range(self.xyz.shape[1])])
        first = np.full(self.S, -1, np.int64)
        first[self.codes[::-1]] = np.arange(len(self.codes))[::-1]
        self.spot_cell = np.where(first >= 0, self.tx_cell[np.maximum(first, 0)], -1)
        self.spot_key = self.spot_cell * 2 + self.side.astype(int)
        self.seed, self.n_jobs = int(seed), int(n_jobs)
        self.k = smooth_neighbours(np.median(self.size[self.size > 0]), smooth_transcripts)
        self.band = None  # 胞质 MetaSpot 的径向档（set_radial_bands）；None = 不去径向趋势

    def set_radial_bands(self, radial: np.ndarray, n_bands: int) -> None:
        """胞质 MetaSpot 按到细胞边界的归一化距离（``radial``，与 spot_names 对齐）分 ``n_bands`` 个等人数档。"""
        r = np.asarray(radial, np.float64)
        cyto = ~self.side
        fill = np.nanmedian(r[cyto]) if np.isfinite(r[cyto]).any() else 0.5
        r = np.where(np.isfinite(r), r, fill)
        edges = np.quantile(r[cyto], np.linspace(0, 1, int(n_bands) + 1)[1:-1])
        self.band = np.searchsorted(edges, r, side="right")

    def sample_transcripts(self, cells: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(self.tx_cell, cells))

    def shuffled_genes(self, rows: np.ndarray) -> np.ndarray:
        """Gene codes of ``rows`` with identities shuffled within each cell side."""
        key = self.tx_cell[rows] * 2 + self.part[rows].astype(int)
        codes = self.gene_codes[rows]
        out = codes.copy()
        rng = np.random.default_rng(self.seed + 11)
        out[np.lexsort((rng.random(len(rows)), key))] = codes[np.argsort(key, kind="stable")]
        return out

    def ge_blocks(self, rows: np.ndarray, gene_codes: np.ndarray, ge: Mapping[str, Any], drop_genes: Optional[np.ndarray] = None) -> list:
        """Raw per-scale MetaSpot vectors from the transcripts ``rows``; genes flagged in ``drop_genes`` are not neighbours."""
        present, (blocks,) = self.ge_blocks_many(rows, ge, [(gene_codes, drop_genes)])
        return self.expand(present, blocks)

    def ge_blocks_many(self, rows: np.ndarray, ge: Mapping[str, Any], variants: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]]):
        """ge_blocks 的多变体版：variants = [(每行基因编号, 不作邻居的基因或 None), ...]。每个细胞的邻居对只找一次。
        只算 ``rows`` 所在的 MetaSpot（内存不随全部数据增长）：返回 (这些 MetaSpot 的编号, [[每尺度数组] 每变体])，用 expand 放回全长。"""
        kept = pd.Index(self.gene_names).get_indexer(np.asarray(ge["genes"]).astype(str))
        has = [(kept >= 0) if drop is None else ((kept >= 0) & ~drop) for _, drop in variants]
        spot = self.codes[rows]
        present, compact = np.unique(spot[spot >= 0], return_inverse=True)
        local = np.full(len(rows), -1, np.int64)
        local[spot >= 0] = compact
        out, start = [[] for _ in variants], 0
        for dim, radius in zip(np.asarray(ge["scale_dims"]).astype(int), np.asarray(ge["neighbour_radii"], np.float64)):
            emb = np.asarray(ge["vectors"], np.float32)[:, start:start + dim]
            blocks = _metaspot_vectors_many(self.tx_cell[rows], [codes for codes, _ in variants], self.xyz[rows], self.part[rows], local,
                                            [kept[h] for h in has], [emb[h] for h in has], len(self.gene_names), len(present), float(radius), self.n_jobs)
            for o, b in zip(out, blocks):
                o.append(b)
            start += dim
        return present, out

    def expand(self, present: np.ndarray, blocks: list) -> list:
        """把只含部分 MetaSpot 的数组放回 spot_names 全长；其余行为 0，与按全长直接计算相同。"""
        full = []
        for b in blocks:
            f = np.zeros((self.S, b.shape[1]), np.float32)
            f[present] = b
            full.append(f)
        return full


def ge_read(
    ge: Mapping[str, Any],
    markers: Sequence[str],
    real: Sequence[np.ndarray],
    null: Sequence[np.ndarray],
    solvers: Sequence[Any],
    whole: Optional[Sequence[np.ndarray]] = None,
    exclude: Optional[np.ndarray] = None,
    scale_weights: Optional[list] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """(real, null, whole) scores of one structure: per scale the projection on the noise-whitened marker direction,
    standardised on the real sample, weighted by its noise share and added. ``real``, ``null``, ``whole`` hold the MetaSpots
    of one side per scale. None when the structure has no marker in the embedding. ``scale_weights``, when a list,
    receives the weight of each scale (record only)."""
    genes = pd.Series(np.asarray(ge["genes"]).astype(str)).str.upper().to_numpy()
    vectors = np.asarray(ge["vectors"], np.float32)
    keep = np.ones(len(genes), bool) if exclude is None else ~exclude
    total_real, total_null, total_whole, start = 0.0, 0.0, 0.0, 0
    for i, dim in enumerate(np.asarray(ge["scale_dims"]).astype(int)):
        d, _ = direction(vectors[keep, start:start + dim], genes[keep], markers)
        start += dim
        if d is None:
            return None
        d = solvers[i](d)
        x, y = real[i] @ d, null[i] @ d
        w = read_weight(x, y)
        if scale_weights is not None:
            scale_weights.append(float(w))
        mu, sd = float(x.mean()), max(float(x.std()), 1e-9)
        total_real = total_real + w * (x - mu) / sd
        total_null = total_null + w * (y - mu) / sd
        if whole is not None:
            t = whole[i] @ d
            total_whole = total_whole + w * (t - t.mean()) / max(float(t.std()), 1e-9)
    return total_real, total_null, (total_whole if whole is not None else None)


def centre_by_band(score: np.ndarray, band: np.ndarray) -> np.ndarray:
    """每个径向档内减去该档均值：去掉细胞群体共有的由外到内趋势，只留同一径向位置上的差别。"""
    n = int(band.max()) + 1 if band.size else 0
    means = np.bincount(band, weights=score, minlength=n) / np.maximum(np.bincount(band, minlength=n), 1)
    return score - means[band]


def lens_direction(lens: Mapping[str, Any], markers: Sequence[str]) -> Optional[np.ndarray]:
    genes = pd.Series(np.asarray(lens["gene_names"]).astype(str)).str.upper().to_numpy()
    return direction(np.asarray(lens["gene_loadings"], np.float32), genes, markers)[0]


def _split_blocks(block: np.ndarray, ge: Mapping[str, Any]) -> list:
    return np.split(np.asarray(block), np.cumsum(np.asarray(ge["scale_dims"]).astype(int))[:-1], axis=1)


def _zscore(x: np.ndarray, stats: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
    mu, sd = stats if stats is not None else (float(x.mean()), max(float(x.std()), 1e-9))
    return (x - mu) / sd, (mu, sd)


def read_side(
    data: AxisData,
    side: bool,
    structures: Sequence[str],
    marker_sets: Mapping[str, Sequence[str]],
    source: str,
    ge: Optional[Mapping[str, Any]],
    ge_all: Optional[list],
    ge_real: Optional[list],
    ge_null: Optional[list],
    lens: Optional[Mapping[str, Any]],
    z: Optional[np.ndarray],
    in_sample: np.ndarray,
    exclude: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Smoothed scores of one side per structure: ``all`` (every MetaSpot of the side), ``real`` and ``null`` (sampled cells).
    ``weights``, when a dict, receives per structure the read-out weights sqrt(1 - rho) / rho (record only): GE per scale,
    the combined GE read-out, LENS, and LENS's share of GE + LENS."""
    mask = data.side == side
    sampled = mask & in_sample
    null_row = None
    if "lens" in source:
        null_row = pd.Series(np.arange(len(lens["null_spots"])), index=np.asarray(lens["null_spots"]).astype(np.int64)).reindex(np.flatnonzero(sampled)).to_numpy()
    real_s = null_s = whole_s = solvers = None
    if "ge" in source:
        real_s = [b[sampled] for b in ge_real]
        null_s = [b[sampled] for b in ge_null]
        whole_s = None if ge_all is None else [b[mask] for b in ge_all]
        solvers = [whitener(b, seed=data.seed) for b in null_s]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    # 近邻只依赖位置：每侧建一次，所有结构的真实 / 打乱 / 全部分数共用
    near_sampled = smooth_neighbours_of(data.centroid[sampled], data.spot_key[sampled], data.k)
    near_all = smooth_neighbours_of(data.centroid[mask], data.spot_key[mask], data.k) if (ge_all is not None or "lens" in source) else None
    for name in structures:
        markers = list(marker_sets.get(name, ()))
        total_real, total_null, total_all = 0.0, 0.0, 0.0
        record: Dict[str, Any] = {}
        if "ge" in source:
            scale_w: list = []
            read = ge_read(ge, markers, real_s, null_s, solvers, whole_s, exclude, scale_weights=scale_w)
            if read is None:
                continue
            x, y, t = read
            z_x, stats = _zscore(x)
            z_y = _zscore(y, stats)[0]
            w = read_weight(z_x, z_y)
            record.update(ge_scale_weights=scale_w, ge_weight=float(w))
            total_real, total_null = w * z_x, w * z_y
            if t is not None:
                total_all = w * _zscore(t)[0]
        if "lens" in source:
            d = lens_direction(lens, markers)
            if d is None:
                continue
            z_x, stats = _zscore(z[sampled] @ d)
            z_y = _zscore(np.asarray(lens["null_z"], np.float32)[null_row] @ d, stats)[0]
            w = read_weight(z_x, z_y)
            record["lens_weight"] = float(w)
            total_real, total_null = total_real + w * z_x, total_null + w * z_y
            total_all = total_all + w * _zscore(z[mask] @ d)[0]
        if weights is not None:
            if "ge_weight" in record and "lens_weight" in record:
                record["lens_share"] = record["lens_weight"] / max(record["ge_weight"] + record["lens_weight"], 1e-12)
            weights[name] = record
        out[name] = {
            "real": smooth_with(np.asarray(total_real), near_sampled),
            "null": smooth_with(np.asarray(total_null), near_sampled),
        }
        if near_all is not None:
            out[name]["all"] = smooth_with(np.asarray(total_all), near_all)
        if data.band is not None and not side:  # 真实、打乱、全部各自按自己的档均值居中
            for key, rows in (("real", sampled), ("null", sampled), ("all", mask)):
                if key in out[name]:
                    out[name][key] = centre_by_band(out[name][key], data.band[rows])
    return out


def assign_labels(scores: Mapping[str, np.ndarray], q: float, broad: str, n: int) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Label per MetaSpot of a side: the structure furthest above its threshold, else ``broad``. ``scores`` holds
    {name: {real, null, all}}. Returns labels and, per structure, threshold, per-MetaSpot false share and share labelled."""
    labels = np.full(n, broad, dtype=object)
    best = np.full(n, -np.inf)
    info: Dict[str, Dict[str, Any]] = {}
    for name, s in scores.items():
        grid, curve = false_share_curve(s["real"], s["null"])
        tau = threshold_for(s["real"], s["null"], q)
        info[name] = {"threshold": tau, "false_share": np.interp(s["all"], grid, curve, left=1.0), "detected": tau is not None}
        if tau is None:
            continue
        excess = (s["all"] - tau) / max(float(np.std(s["null"])), 1e-9)
        hit = (excess > 0) & (excess > best)
        labels[hit] = name
        best[hit] = excess[hit]
    return labels, info


def validate(
    data: AxisData,
    sides: Mapping[bool, Sequence[str]],
    marker_sets: Mapping[str, Sequence[str]],
    ge: Mapping[str, Any],
    cells: np.ndarray,
    q: float,
    *,
    folds: int,
    min_gene_transcripts: int,
    n_boot: int,
    support_alpha: float,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, Any]]:
    """Held-out marker enrichment of the labels each structure would get (gene-embedding read-out, report only).

    Every gene is held out once. In its fold it is neither a neighbour in the MetaSpot vectors nor part of a marker
    direction, so its own transcripts cannot lift the score at their position (gene vectors stay as fitted). The
    expectation for a held-out gene is the share of each cell's remaining transcripts that fall in labelled units.
    """
    from .evidence_annotation import _marker_gene_enrichment, _median_enrichment

    genes = pd.Series(np.asarray(ge["genes"]).astype(str)).str.upper().to_numpy()
    all_markers = {name: np.isin(genes, list(marker_sets.get(name, ()))) for side in sides.values() for name in side}
    any_marker = np.logical_or.reduce(list(all_markers.values())) if all_markers else np.zeros(len(genes), bool)
    fold_of = np.full(len(genes), -1)
    for group in [np.flatnonzero(m) for m in all_markers.values()] + [np.flatnonzero(~any_marker)]:
        group = group[fold_of[group] < 0]
        fold_of[rng.permutation(group)] = np.arange(group.size) % folds
    rows = data.sample_transcripts(cells)
    in_sample = np.isin(data.spot_cell, cells)
    shuffled = data.shuffled_genes(rows)
    gene_of_row = pd.Series(data.gene_names).str.upper().to_numpy()[data.gene_codes[rows]]
    column = pd.Series(np.arange(len(genes)), index=genes).reindex(gene_of_row).to_numpy()
    usable = np.isfinite(column) & (data.codes[rows] >= 0)
    counts = sp.csr_matrix((np.ones(int(usable.sum())), (data.codes[rows][usable], column[usable].astype(np.int64))), shape=(data.S, len(genes)))
    keys = {side: {name: {"observed": np.zeros(len(genes)), "expected": np.zeros(len(genes))} for name in names} for side, names in sides.items()}
    # 所有折的真实 / 打乱向量一次算完：邻居对只依赖位置，每个细胞只找一次
    variants = [(codes, fold_of == fold) for fold in range(int(folds)) for codes in (data.gene_codes[rows], shuffled)]
    present, blocks = data.ge_blocks_many(rows, ge, variants)
    for fold in range(int(folds)):
        held = fold_of == fold
        real, null = data.expand(present, blocks[2 * fold]), data.expand(present, blocks[2 * fold + 1])
        for side, names in sides.items():
            scores = read_side(data, side, names, marker_sets, "ge", ge, None, real, null, None, None, in_sample, exclude=held)
            sampled = np.flatnonzero((data.side == side) & in_sample)
            X = counts[sampled]
            cell = np.unique(data.spot_cell[sampled], return_inverse=True)[1]
            cell_gene = (sp.csr_matrix((np.ones(len(cell)), (cell, np.arange(len(cell)))), shape=(cell.max() + 1, len(cell))) @ X).tocsr()
            n_train = np.asarray(X[:, ~held].sum(axis=1)).ravel()
            size = np.maximum(np.bincount(cell, weights=n_train), 1.0)
            for name, s in scores.items():
                tau = threshold_for(s["real"], s["null"], q)
                called = (s["real"] > tau).astype(np.float64) if tau is not None else np.zeros(len(sampled))
                share = np.bincount(cell, weights=n_train * called, minlength=len(size)) / size
                keys[side][name]["observed"][held] = np.asarray(X[:, held].T @ called).ravel()
                keys[side][name]["expected"][held] = np.asarray(cell_gene[:, held].T @ share).ravel()
    out: Dict[str, Dict[str, Any]] = {}
    for side, names in sides.items():
        sub = counts[np.flatnonzero((data.side == side) & in_sample)]
        eligible = np.asarray(sub.sum(axis=0)).ravel() >= int(min_gene_transcripts)
        for name in names:
            seed = all_markers[name] & eligible
            rivals = {o: all_markers[o] & eligible & ~seed for o in names if o != name}
            taken = np.logical_or.reduce(list(rivals.values())) if rivals else np.zeros(len(genes), bool)
            background = eligible & ~seed & ~taken
            k = keys[side][name]
            res = _marker_gene_enrichment(k["observed"], k["expected"], seed, background, n_boot=n_boot, rng=rng, support_alpha=support_alpha)
            rival = {o: _median_enrichment(k["observed"], k["expected"], m, background) for o, m in rivals.items()}
            res.update(
                competitor_enrichment=rival,
                specific=bool(np.isfinite(res["marker_enrichment"]) and all(not np.isfinite(v) or res["marker_enrichment"] > v for v in rival.values())),
                supported=bool(np.isfinite(res["support_bound"]) and res["support_bound"] > 1.0),
            )
            out[name] = res
    return out


def annotate(
    transcripts: pd.DataFrame,
    inputs: Mapping[str, Any],
    sides: Mapping[bool, Sequence[str]],
    broad: Mapping[bool, str],
    marker_sets: Mapping[str, Sequence[str]],
    options: Mapping[str, Any],
    *,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Scores, labels and held-out report of every structure. ``inputs``: spot_names, and ge + ge_block and/or lens + z.

    Returns one row per MetaSpot of ``spot_names``: label, per structure ``score__`` and ``q__``; and the report."""
    source = str(options.get("evidence_axis_source", "auto")).lower()
    has_lens = inputs.get("lens") is not None and inputs.get("z") is not None and len(inputs["lens"].get("null_spots", ())) > 0
    if source == "auto":
        source = "ge+lens" if has_lens else "ge"
    if source not in {"ge", "lens", "ge+lens"}:
        raise ValueError("annotation.evidence_axis_source must be 'auto', 'ge', 'lens' or 'ge+lens'")
    if "lens" in source and not has_lens:
        raise ValueError(f"annotation.evidence_axis_source={source!r} needs the masked-transcript model with lens_null_cells > 0")
    ge, lens = inputs.get("ge"), inputs.get("lens")
    if "ge" in source and ge is None:
        raise ValueError(f"annotation.evidence_axis_source={source!r} needs the spatial gene embedding")
    data = AxisData(transcripts, inputs["spot_names"], seed=int(options.get("evidence_seed", 0)), n_jobs=n_jobs,
                    smooth_transcripts=float(options.get("evidence_axis_smooth_transcripts", AXIS_SMOOTH_TRANSCRIPTS)))
    q = float(options.get("evidence_axis_q", 0.2))
    radial_bands = int(options.get("evidence_axis_radial_bands", 0) or 0)
    if radial_bands > 0:
        if inputs.get("radial") is None:
            raise ValueError("annotation.evidence_axis_radial_bands needs the MetaSpot distance to the cell edge (adata1.obs d_cell_norm_mean)")
        data.set_radial_bands(inputs["radial"], radial_bands)
    rng = np.random.default_rng(int(options.get("evidence_seed", 0)))
    n_cells = int(data.spot_cell.max()) + 1
    max_cells = int(options.get("evidence_max_cells", 10000))
    if "lens" in source:  # the null of the read-out through W exists for the cells the model encoded
        cells = np.unique(data.spot_cell[np.asarray(lens["null_spots"]).astype(np.int64)])
    elif max_cells > 0 and n_cells > max_cells:
        cells = np.sort(rng.choice(n_cells, max_cells, replace=False))
    else:
        cells = np.arange(n_cells)
    in_sample = np.isin(data.spot_cell, cells)
    ge_real = ge_null = ge_all = None
    if "ge" in source:
        rows = data.sample_transcripts(cells)
        present, (real_blocks, null_blocks) = data.ge_blocks_many(rows, ge, [(data.gene_codes[rows], None), (data.shuffled_genes(rows), None)])
        ge_real, ge_null = data.expand(present, real_blocks), data.expand(present, null_blocks)
        ge_all = _split_blocks(inputs["ge_block"], ge)
    z = None if inputs.get("z") is None else np.asarray(inputs["z"], np.float32)
    table = pd.DataFrame({"metaspot": data.names})
    table["label"] = np.where(data.side, broad[True], broad[False])
    report: Dict[str, Dict[str, Any]] = {}
    for side, names in sides.items():
        mask = data.side == side
        if mask.sum() < 10:
            continue
        read_weights: Dict[str, Dict[str, Any]] = {}
        scores = read_side(data, side, names, marker_sets, source, ge, ge_all, ge_real, ge_null, lens, z, in_sample,
                           weights=read_weights)
        labels, info = assign_labels(scores, q, broad[side], int(mask.sum()))
        table.loc[mask, "label"] = labels
        for name in names:
            found = name in scores
            used = int(np.isin(pd.Series(np.asarray(ge["genes"]).astype(str)).str.upper(), list(marker_sets.get(name, ()))).sum()) if ge is not None else (
                int(np.isin(pd.Series(np.asarray(lens["gene_names"]).astype(str)).str.upper(), list(marker_sets.get(name, ()))).sum()))
            table[f"score__{name}"] = table.get(f"score__{name}", np.nan)
            table[f"q__{name}"] = table.get(f"q__{name}", np.nan)
            if found:
                table.loc[mask, f"score__{name}"] = scores[name]["all"]
                table.loc[mask, f"q__{name}"] = info[name]["false_share"]
            report[name] = {"marker_genes_used": used, "levels": {}, "level": "axis", "rank_level": "axis", "supported": False, "suggestive": False,
                            "rank_supported": False, "detected": bool(found and info[name]["detected"]), "axis_threshold": info[name]["threshold"] if found else None,
                            "axis_source": source, "axis_false_share_target": q, "units_called": int((labels == name).sum()),
                            "axis_weights": read_weights.get(name, {})}
    if ge is not None and int(options.get("evidence_validation_folds", 5)) > 1:
        valid_cells = cells
        cap = int(options.get("evidence_axis_validation_cells", 3000))
        if cap > 0 and len(valid_cells) > cap:
            valid_cells = np.sort(np.random.default_rng(int(options.get("evidence_seed", 0)) + 1).choice(valid_cells, cap, replace=False))
        n_tests = sum(1 for names in sides.values() for name in names if report.get(name, {}).get("marker_genes_used"))
        checked = {side: [n for n in names if report.get(n, {}).get("marker_genes_used", 0) >= max(4, int(options.get("evidence_validation_folds", 5)))] for side, names in sides.items()}
        res = validate(data, checked, marker_sets, ge, valid_cells, q, folds=int(options.get("evidence_validation_folds", 5)),
                       min_gene_transcripts=int(options.get("evidence_min_gene_transcripts", 50)), n_boot=int(options.get("evidence_bootstrap", 2000)),
                       support_alpha=0.05 / max(n_tests, 1), rng=rng)
        for name, v in res.items():
            report[name]["levels"] = {"axis": v}
            report[name]["supported"] = bool(v["supported"])
            report[name]["rank_supported"] = bool(v["supported"])
            report[name]["suggestive"] = bool(v["supported"] or (np.isfinite(v["ci_low"]) and v["ci_low"] > 1.0))
    return table, report
