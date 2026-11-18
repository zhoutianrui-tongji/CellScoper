"""MetaDomain discovery from the masked-transcript model (LENS): structure types found by density against a null.

A MetaDomain is a direction of the MetaSpot embedding z that real MetaSpots take far more often than MetaSpots of cells
whose gene identities were shuffled within each cell side (``null_z``: same model, same positions, no gene placement).
The number of MetaDomains comes from the data: only density modes of real MetaSpots that beat the null are kept, and
MetaSpots that do not clearly belong to one stay background. Reference markers are not used here; they name MetaDomains
afterwards through the decoder (``domain_gene_profiles``).

Nucleus and cytoplasm MetaSpots are clustered together (``split_sides=True``: each side alone, the 2026-11-02 version):
  1 smooth z over neighbouring MetaSpots of the same cell side (as the structure axes), real and null alike
  2 whiten by the null covariance: variation the shuffled cells also show (positions, depth, noise) is down-weighted
  3 detect MetaSpots whose whitened norm exceeds the null at false share ``q``; thresholds per stratum = side x depth
    quartile of the cell side, so shallow cells (shorter z) are not judged against deep ones and each side against its own
    null (calibration only: the side never splits a MetaDomain)
  4 density clustering (HDBSCAN) of the detected MetaSpots' unit directions (PCA, at most ``max_fit`` MetaSpots, equal
    numbers from each side): each mode gives one candidate direction; noise points are not forced into any mode
  5 every candidate direction becomes an axis: projection of the whitened z, threshold from the null at false share
    ``q`` per stratum; a MetaSpot takes the MetaDomain whose threshold it exceeds most (ratio), else background
  6 reproducibility: steps 4 on two disjoint halves of the cells; a candidate is kept when both halves have a mutual-best
    direction with cosine >= ``min_repro``
  7 consensus over restarts: steps 4 and 6 are repeated ``n_restarts`` times with different samples and cell halves; the
    directions kept in each restart are grouped (cosine >= ``min_repro``), and a group kept in at least half of the restarts
    becomes one MetaDomain (mean direction). One restart depends on its 20k-MetaSpot sample: on Atera 5k (2026-11-03) the
    same z gave 0-4 MetaDomains across five seeds, while the 4 found when it worked were the same with both LENS seeds
    (decoder profile r .87-.94; scripts/mdd_seed_diag.py)

Why together (2026-11-03): z is relative to each cell side, and the nucleus flag comes from a 2D outline. Clustered per
side, Atera 5k gave mirror pairs across the outline (nuclear-edge MetaDomain with cytoplasm-like genes, r = .93 with the
peripheral cytoplasm one; perinuclear cytoplasm MetaDomains with nuclear-like genes, r = .70-.79 with the nuclear interior
one), and in the raw data RNA composition hardly differs across the outline (within-cell AUC of the nucleus flag .51-.60
next to it) while transcripts at the top and bottom of the outline look cytoplasmic (scripts/nucleus_gene_info.py).
``nucleus_boundary_check`` reports, per MetaDomain, how far its genes follow the nucleus-cytoplasm contrast of the raw
data and how many of its MetaSpots lie on the opposite side of the outline.

Atera BRCA 1k (2026-11-02, scripts metadomain_discovery.py / mdd_compare.py, prototype of this module): gene-shuffled
input gives no MetaDomain; 3-6 nuclear and 2-9 cytoplasmic MetaDomains per run; matched between runs with separately
trained models for 76-83% of the clustered MetaSpots (KMeans 25 per side: 20-26%); without markers, the MetaDomain whose
decoder profile ranks the reference markers highest holds held-out ER genes at OR 1.92 (KMeans 1.42) and Speckle genes at
1.66 (1.45).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .lens import _smooth
from .structure_axes import text_codes, threshold_for

WHITEN_RIDGE = 1e-3     # ridge of the null covariance, share of its mean eigenvalue
N_PCA = 10              # dimensions of the direction space for density clustering
MIN_CLUSTER_SHARE = 0.005  # smallest density mode, as a share of the side's detected MetaSpots
MIN_CLUSTER = 30
MIN_DETECTED = 50       # fewer detected MetaSpots on a side: no MetaDomains there
N_RESTARTS = 5          # independent sample + cell-half draws whose reproducible directions are combined (step 7)


@dataclass
class GroupResult:
    """Discovery on one group of MetaSpots (both sides, or one). ``domain``: per MetaSpot of the group, index into ``directions`` or -1 (background)."""
    directions: np.ndarray          # domains x d, unit vectors in whitened z space
    whitener: np.ndarray            # d x d
    domain: np.ndarray              # per MetaSpot of the group
    score: np.ndarray               # projection on its domain's direction / that domain's threshold (>= 1 inside), 0 outside
    detected: np.ndarray            # bool, whitened norm above the null threshold
    report: List[Dict[str, float]]  # one dict per domain


def whiten_by(null_rows: np.ndarray, ridge: float = WHITEN_RIDGE) -> np.ndarray:
    """Symmetric inverse square root of the null covariance (plus a ridge)."""
    cov = np.atleast_2d(np.cov(np.asarray(null_rows, np.float64).T))
    cov = cov + ridge * np.trace(cov) / len(cov) * np.eye(len(cov))
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def depth_bins(depth_real: np.ndarray, n_bins: int = 4) -> np.ndarray:
    edges = np.quantile(depth_real, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def strata_codes(depth_real: np.ndarray, depth_null: np.ndarray, side_real: np.ndarray, side_null: np.ndarray,
                 n_bins: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Stratum of every real and null MetaSpot: side x depth quartile (edges from the real MetaSpots of that side)."""
    code_real, code_null = np.zeros(len(depth_real), np.int64), np.zeros(len(depth_null), np.int64)
    for s in (False, True):
        mr, mn = side_real == s, side_null == s
        if not mr.any():
            continue
        inner = depth_bins(depth_real[mr], n_bins)[1:-1]
        code_real[mr] = int(s) * n_bins + np.searchsorted(inner, depth_real[mr], side="left")
        code_null[mn] = int(s) * n_bins + np.searchsorted(inner, depth_null[mn], side="left")
    return code_real, code_null


def binned_thresholds(real: np.ndarray, null: np.ndarray, code_real: np.ndarray, code_null: np.ndarray,
                      q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-MetaSpot threshold (inf where the null is never beaten) for real and null MetaSpots, one per stratum."""
    t_real, t_null = np.full(len(real), np.inf), np.full(len(null), np.inf)
    for c in np.unique(code_real):
        mr, mn = code_real == c, code_null == c
        if mr.sum() < 10 or mn.sum() < 10:
            continue
        tau = threshold_for(real[mr], null[mn], q)
        if tau is not None:
            t_real[mr], t_null[mn] = tau, tau
    return t_real, t_null


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def density_directions(u: np.ndarray, *, max_fit: int, seed: int) -> np.ndarray:
    """Mean unit direction of every HDBSCAN density mode of the unit vectors ``u`` (fit on at most ``max_fit`` rows)."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    if len(u) > max_fit:
        u = u[np.random.default_rng(seed).choice(len(u), max_fit, replace=False)]
    x = PCA(n_components=min(N_PCA, u.shape[1]), random_state=seed).fit_transform(u)
    min_size = max(MIN_CLUSTER, int(MIN_CLUSTER_SHARE * len(u)))
    labels = HDBSCAN(min_cluster_size=min_size, min_samples=10).fit(x).labels_
    if labels.max() < 0:
        # HDBSCAN 默认不允许整棵树只有一个类：数据里只有一种结构时会全判为噪声；此时退回单类模式取它的方向
        # （只用方向，成员由后面的轴分配决定）
        labels = HDBSCAN(min_cluster_size=min_size, min_samples=10, allow_single_cluster=True).fit(x).labels_
    if labels.max() < 0:
        return np.zeros((0, u.shape[1]))
    return _unit_rows(np.array([u[labels == k].mean(0) for k in range(labels.max() + 1)]))


def mutual_best_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row of ``a``: cosine with its best row of ``b`` when that row's best is also this row, else nan."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.nan)
    cos = a @ b.T
    best_b, best_a = cos.argmax(1), cos.argmax(0)
    return np.array([cos[i, best_b[i]] if best_a[best_b[i]] == i else np.nan for i in range(len(a))])


def balanced_sample(group: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Row indices: at most ``size`` rows in total, the same number from each group value (fewer if a group is smaller)."""
    values = np.unique(group)
    per = max(1, size // max(len(values), 1))
    rng = np.random.default_rng(seed)
    rows = [rng.permutation(np.flatnonzero(group == v))[:per] for v in values]
    return np.sort(np.concatenate(rows)) if rows else np.zeros(0, np.int64)


def reproducible_directions(u: np.ndarray, u_side: np.ndarray, u_cell: np.ndarray, *, min_repro: float, max_fit: int,
                            seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One restart of steps 4 and 6: density directions of a balanced sample of the unit vectors ``u`` that a mutual-best
    direction reproduces in both halves of the cells. Returns (directions, cosine in half 1, cosine in half 2)."""
    fit = balanced_sample(u_side, max_fit, seed)
    candidates = density_directions(u[fit], max_fit=max_fit, seed=seed)
    if not len(candidates):
        return np.zeros((0, u.shape[1])), np.zeros(0), np.zeros(0)
    half = np.random.default_rng(seed + 1).random(int(u_cell.max()) + 1) < 0.5
    in_half = half[u_cell]
    repro = []
    for i, h in enumerate((in_half, ~in_half)):
        rows = np.flatnonzero(h)
        sub = rows[balanced_sample(u_side[rows], max_fit, seed + 2 + i)]
        repro.append(mutual_best_cosine(candidates, density_directions(u[sub], max_fit=max_fit, seed=seed + 2 + i)))
    keep = np.flatnonzero(np.nan_to_num(np.minimum(repro[0], repro[1]), nan=-1.0) >= min_repro)
    return candidates[keep], repro[0][keep], repro[1][keep]


def consensus_directions(runs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], dim: int, *, min_repro: float,
                         min_support: float = 0.5) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Step 7: group the directions kept in each restart (greedy, cosine >= ``min_repro`` with the group's mean direction,
    at most one direction per restart in a group) and keep groups found in at least ``min_support`` of the restarts.
    Returns the groups' mean unit directions and, per group, support and mean half-cell cosines."""
    groups: List[Dict[str, object]] = []
    for r, (dirs, c1, c2) in enumerate(runs):
        for d, a, b in zip(dirs, c1, c2):
            best, best_cos = None, min_repro
            for g in groups:
                if r in g["runs"]:
                    continue
                cos = float(_unit_rows(np.asarray(g["sum"])[None])[0] @ d)
                if cos >= best_cos:
                    best, best_cos = g, cos
            if best is None:
                groups.append({"sum": d.copy(), "runs": {r}, "c1": [a], "c2": [b]})
            else:
                best["sum"] = best["sum"] + d
                best["runs"].add(r); best["c1"].append(a); best["c2"].append(b)
    need = max(1, int(np.ceil(min_support * len(runs))))
    kept = [g for g in groups if len(g["runs"]) >= need]
    if not kept:
        return np.zeros((0, dim)), []
    directions = _unit_rows(np.array([g["sum"] for g in kept]))
    report = [{"support": len(g["runs"]) / len(runs), "repro_half1": float(np.mean(g["c1"])),
               "repro_half2": float(np.mean(g["c2"]))} for g in kept]
    return directions, report


def discover_group(z_g: np.ndarray, null_g: np.ndarray, code_real: np.ndarray, code_null: np.ndarray,
                   side_g: np.ndarray, cell_g: np.ndarray, *, q: float = 0.2, min_repro: float = 0.8,
                   max_fit: int = 20000, seed: int = 0, n_restarts: int = N_RESTARTS) -> GroupResult:
    """Steps 2-7 on one group of MetaSpots (both sides, or one); ``z_g`` and ``null_g`` are already smoothed.
    ``code_real`` / ``code_null``: threshold strata (``strata_codes``); ``side_g``: nucleus flag per MetaSpot (balances the
    clustering sample); ``cell_g``: cell index per MetaSpot; ``n_restarts``: restarts combined in step 7."""
    Wh = whiten_by(null_g)
    zw, nw = z_g @ Wh, null_g @ Wh
    r_real, r_null = (zw ** 2).sum(1), (nw ** 2).sum(1)
    t_real, _ = binned_thresholds(r_real, r_null, code_real, code_null, q)
    detected = r_real >= t_real
    empty = GroupResult(np.zeros((0, zw.shape[1])), Wh, np.full(len(zw), -1), np.zeros(len(zw)), detected, [])
    if detected.sum() < MIN_DETECTED:
        return empty
    det_rows = np.flatnonzero(detected)
    u, u_side, u_cell = _unit_rows(zw[det_rows]), side_g[det_rows], cell_g[det_rows]
    # 每次重启的样本和细胞分半都不同（种子间隔 10，避免与上一轮的 seed + 2/3 重叠）；只保留多数重启都复现的方向
    runs = [reproducible_directions(u, u_side, u_cell, min_repro=min_repro, max_fit=max_fit, seed=seed + 10 * r)
            for r in range(max(1, int(n_restarts)))]
    directions, consensus = consensus_directions(runs, u.shape[1], min_repro=min_repro)
    if not len(directions):
        return empty
    # each kept direction as an axis with its own null threshold; a MetaSpot takes the domain it exceeds most
    ratio = np.zeros((len(zw), len(directions)))
    report = []
    for j, d in enumerate(directions):
        s_real, s_null = zw @ d, nw @ d
        t_r, t_n = binned_thresholds(s_real, s_null, code_real, code_null, q)
        ratio[:, j] = np.where(np.isfinite(t_r) & (t_r > 0), s_real / np.where(np.isfinite(t_r) & (t_r > 0), t_r, 1.0), 0.0)
        report.append({"candidate": j, **consensus[j],
                       "null_above": float(np.mean(s_null >= t_n)), "real_above": float(np.mean(s_real >= t_r))})
    if len(directions):
        best = ratio.argmax(1)
        top = ratio[np.arange(len(zw)), best]
        domain = np.where(top >= 1.0, best, -1)
        score = np.where(domain >= 0, top, 0.0)
        # a direction that wins no MetaSpot (always exceeded more by a neighbouring one) is dropped
        used = np.flatnonzero(np.bincount(domain[domain >= 0], minlength=len(directions)) > 0)
        remap = np.full(len(directions), -1)
        remap[used] = np.arange(len(used))
        domain = np.where(domain >= 0, remap[np.maximum(domain, 0)], -1)
        directions, report = directions[used], [report[j] for j in used]
    else:
        domain, score = np.full(len(zw), -1), np.zeros(len(zw))
    return GroupResult(directions, Wh, domain, score, detected, report)


def discover_metadomains(z: np.ndarray, null_z: np.ndarray, null_rows: np.ndarray, nuclear: np.ndarray, spot_key: np.ndarray,
                         centroid: np.ndarray, side_depth: np.ndarray, spot_cell: np.ndarray, *, smooth_k: int = 5,
                         q: float = 0.2, min_repro: float = 0.8, max_fit: int = 20000, seed: int = 0,
                         split_sides: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[object, GroupResult], np.ndarray]:
    """MetaDomains of all MetaSpots. ``null_rows``: MetaSpot row of each ``null_z`` row. ``side_depth``: transcripts of each
    MetaSpot's cell side. ``split_sides``: cluster each side alone (groups True / False) instead of both together (group
    "both"). Returns (domain id per MetaSpot, -1 = background; score; one row per domain; per-group results; smoothed z).
    Domain ids run from the most nuclear MetaDomain to the most cytoplasmic one. Table columns: side (nucleus / cytoplasm
    when >= 80% of the members are on that side, else both), nucleus_share of the members, share_of_group (members over the
    group's MetaSpots: all MetaSpots when the sides are clustered together), cells, cell_share, report of ``discover_group``."""
    null_rows = np.asarray(null_rows, np.int64)
    nuclear = np.asarray(nuclear, bool)
    zs = _smooth(np.asarray(z, np.float32), centroid, spot_key, int(smooth_k))
    ns = _smooth(np.asarray(null_z, np.float32), centroid[null_rows], spot_key[null_rows], int(smooth_k))
    domain, score = np.full(len(z), -1), np.zeros(len(z))
    groups = [(True, nuclear), (False, ~nuclear)] if split_sides else [("both", np.ones(len(z), bool))]
    rows, results, offset = [], {}, 0
    for key, in_group in groups:
        real = np.flatnonzero(in_group)
        nm = in_group[null_rows]
        if len(real) < MIN_DETECTED or nm.sum() < MIN_DETECTED:
            continue
        code_real, code_null = strata_codes(side_depth[real], side_depth[null_rows][nm], nuclear[real], nuclear[null_rows][nm])
        res = discover_group(zs[real], ns[nm], code_real, code_null, nuclear[real], spot_cell[real], q=q,
                             min_repro=min_repro, max_fit=max_fit, seed=seed)
        # 编号：核内比例从高到低
        nuc_share = np.array([nuclear[real[res.domain == j]].mean() for j in range(len(res.directions))])
        order = np.argsort(-nuc_share, kind="stable")
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        if len(order):
            res.domain = np.where(res.domain >= 0, rank[np.maximum(res.domain, 0)], -1)
            res.directions, res.report = res.directions[order], [res.report[j] for j in order]
        results[key] = res
        inside = res.domain >= 0
        domain[real[inside]] = res.domain[inside] + offset
        score[real] = res.score
        group_cells = max(np.unique(spot_cell[real]).size, 1)
        for j, rep in enumerate(res.report):
            members = real[res.domain == j]
            share_nuc = float(nuclear[members].mean())
            rows.append({"metadomain": offset + j,
                         "side": "nucleus" if share_nuc >= 0.8 else ("cytoplasm" if share_nuc <= 0.2 else "both"),
                         "nucleus_share": share_nuc, "metaspots": int(len(members)), "share_of_group": len(members) / len(real),
                         "cells": int(np.unique(spot_cell[members]).size),
                         "cell_share": np.unique(spot_cell[members]).size / group_cells, **rep})
        offset += len(res.directions)
    return domain, score, pd.DataFrame(rows), results, zs


def gene_nuclear_log_odds(cell: np.ndarray, gene: np.ndarray, nuclear: np.ndarray) -> pd.Series:
    """Nucleus-cytoplasm contrast of every gene in the raw data: log odds ratio of its transcripts being inside the nucleus
    outline, against the number expected from each transcript's cell (its nuclear share). Index: upper-case gene names."""
    # 按整数编号分组（与按文字分组的组和行序相同，结果逐位一致）；大写只对唯一基因名做一次
    gene_codes, gene_text = text_codes(gene if isinstance(gene, pd.Series) else pd.Series(np.asarray(gene)))
    upper_id, upper_names = pd.factorize(pd.Index(gene_text).str.upper().to_numpy())
    cell_codes = text_codes(cell if isinstance(cell, pd.Series) else pd.Series(np.asarray(cell)))[0]
    t = pd.DataFrame({"cell": cell_codes, "gene": upper_id[gene_codes], "y": np.asarray(nuclear, bool)})
    t["p"] = t.groupby("cell").y.transform("mean")
    g = t.groupby("gene").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    g.index = pd.Index(np.asarray(upper_names)[g.index.to_numpy()].astype(str), name="gene")
    g = g.sort_index()
    return np.log((g.obs + 1) / (g.exp + 1)) - np.log((g.n - g.obs + 1) / (g.n - g.exp + 1))


def nucleus_boundary_check(profiles: np.ndarray, genes: Sequence[str], gene_log_odds: pd.Series, domain: np.ndarray,
                           nuclear: np.ndarray, *, r_min: float = 0.5, opposite_min: float = 0.2) -> pd.DataFrame:
    """Per MetaDomain: ``nucleus_axis_r`` = correlation over genes of its decoder profile with the raw nucleus-cytoplasm
    contrast (``gene_nuclear_log_odds``; > 0 nuclear-like genes); ``opposite_share`` = share of its MetaSpots on the side of
    the nucleus outline its genes do not belong to (outside for nuclear-like, inside for cytoplasm-like);
    ``nucleus_boundary_flag`` when |r| >= ``r_min`` and opposite_share >= ``opposite_min``: the MetaDomain follows the
    nucleus-cytoplasm contrast across the outline (2D outline, segmentation, or a real perinuclear gradient; not a separate
    structure by itself). A nucleus-only MetaDomain with nuclear-like genes (e.g. speckle-like) is not flagged."""
    lo = pd.Series(gene_log_odds).reindex(pd.Series(np.asarray(genes)).astype(str).str.upper().to_numpy()).to_numpy()
    ok = np.isfinite(lo)
    out = []
    for k, prof in enumerate(np.asarray(profiles)):
        m = ok & np.isfinite(prof)
        r = float(np.corrcoef(prof[m], lo[m])[0, 1]) if m.sum() >= 3 else float("nan")
        members = np.asarray(nuclear)[domain == k]
        opposite = float(np.mean(~members)) if r > 0 else float(np.mean(members))
        out.append({"nucleus_axis_r": r, "opposite_share": opposite if len(members) else float("nan"),
                    "nucleus_boundary_flag": bool(np.isfinite(r) and abs(r) >= r_min and opposite >= opposite_min)})
    return pd.DataFrame(out)


def domain_gene_profiles(zs: np.ndarray, domain: np.ndarray, gene_loadings: np.ndarray, n_domains: int,
                         unit_rows: bool = True) -> np.ndarray:
    """Decoder profile of each MetaDomain: W @ mean smoothed z of its members (log-odds of each gene against the cell side).

    With ``unit_rows`` every gene's decoder row is scaled to length 1, so genes rank by how well their row aligns with the
    MetaDomain, not by the size of their loadings (Atera 5k: with raw rows, immunoglobulin genes with large loadings topped
    cytoplasmic MetaDomains whose MetaSpots in non-plasma cells hold almost no immunoglobulin transcripts)."""
    W = np.asarray(gene_loadings, np.float64)
    if unit_rows:
        W = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    out = np.full((n_domains, W.shape[0]), np.nan)
    for k in range(n_domains):
        if np.any(domain == k):
            out[k] = W @ zs[domain == k].mean(0)
    return out


def marker_auc(profile: np.ndarray, genes: Sequence[str], markers: Sequence[str]) -> float:
    """Rank AUC of a structure's reference markers in a gene profile (0.5 = random, nan = fewer than 3 markers)."""
    genes = np.asarray([str(g).upper() for g in genes])
    m = np.isin(genes, [str(g).upper() for g in markers])
    if m.sum() < 3 or not np.isfinite(profile).all():
        return float("nan")
    r = rankdata(profile)
    return float((r[m].mean() - (m.sum() + 1) / 2) / (len(profile) - m.sum()))


def discover_for_adata(adata1, point_df: pd.DataFrame, *, smooth_transcripts: float = 50.0, q: float = 0.2,
                       seed: int = 0, n_top_genes: int = 20) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Pipeline adapter: MetaDomains of ``adata1`` from its LENS embedding (``obsm['X_lens']``, ``uns['lens']``)
    and the transcript table (cell_id, feature_name, x/y[/z]_location, in_nucleus, metaspot). Returns the MetaDomain per
    MetaSpot (-1 = background), its score, and a summary (one entry per MetaDomain: side, nucleus share, size,
    reproducibility, null share, nucleus-boundary check, top genes of the unit-row decoder profile) that h5ad can store."""
    from .structure_axes import AxisData

    spots = adata1.obs_names.astype(str).to_numpy()
    lens = adata1.uns["lens"]
    data = AxisData(point_df, spots, seed=seed, smooth_transcripts=smooth_transcripts)
    side_depth = pd.Series(data.size).groupby(data.spot_key).transform("sum").to_numpy()
    domain, score, table, _, zs = discover_metadomains(
        np.asarray(adata1.obsm["X_lens"], np.float32), np.asarray(lens["null_z"], np.float32), lens["null_spots"],
        data.side, data.spot_key, data.centroid, side_depth, data.spot_cell, smooth_k=data.k, q=q, seed=seed)
    genes = np.asarray(lens["gene_names"]).astype(str)
    profiles = domain_gene_profiles(zs, domain, np.asarray(lens["gene_loadings"]), len(table))
    if len(table):
        gene_lo = gene_nuclear_log_odds(point_df["cell_id"], point_df["feature_name"],
                                        point_df["in_nucleus"].to_numpy(bool))
        table = pd.concat([table, nucleus_boundary_check(profiles, genes, gene_lo, domain, data.side)], axis=1)
    summary = {col: table[col].to_numpy() for col in table.columns} if len(table) else {"metadomain": np.zeros(0, np.int64)}
    summary["top_genes"] = np.array([",".join(genes[np.argsort(-p)[:n_top_genes]]) for p in profiles], dtype=object).astype(str)
    summary["gene_profiles"] = np.asarray(profiles, np.float32)   # MetaDomains x genes, unit-row decoder profile
    summary["gene_names"] = genes
    summary["q"] = np.float64(q)
    return domain, score, summary


__all__ = ["GroupResult", "consensus_directions", "discover_for_adata", "discover_group", "discover_metadomains", "domain_gene_profiles",
           "gene_nuclear_log_odds", "marker_auc", "nucleus_boundary_check", "strata_codes", "whiten_by"]
