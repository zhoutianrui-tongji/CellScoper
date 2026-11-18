"""LENS counting labels (``annotation.evidence_label_rule: lens_count``): labels whose number per cell is the structure's
amount and whose place in the cell follows the structure scores.

Needs LENS trained with the cell's own composition as background (``module5.lens_background: own``): its decoder W then
records which genes sit together inside cells, not which genes cells of one type share. Per structure:

1. Gate. Reference markers are split into GATE_FOLDS parts; W projected on the direction of 4/5 of them must rank the held-out
   fifth above non-reference genes (AUROC). The same is done for GATE_NULL_SETS random gene sets matched to the markers by
   expression decile; the structure is labelled only if z = (AUROC - null mean) / null SD >= GATE_Z. Genes shuffled within
   cells give z near 0.
2. Amount. Gene weight = W projected on the marker direction, divided by the WEIGHT_CAP_QUANTILE quantile of the markers'
   values and clipped to [0, 1]. A MetaSpot's evidence = mean weight of its transcripts, not centred per cell. One Otsu
   threshold per structure over all MetaSpots of its side; above it the MetaSpot counts for the structure (several: the
   largest evidence / threshold). The counted transcripts of a cell side give that structure's quota in the cell.
3. Place. Within each cell side, MetaSpots are ranked by the mean of two percentile ranks, the evidence and the structure-axis
   score (annotation scores; evidence only when a structure has none), and each structure takes MetaSpots from the top until
   its quota is filled; a MetaSpot goes to the structure for which it lies earliest in its quota.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

GATE_Z = 3.0
GATE_NULL_SETS = 30
GATE_FOLDS = 5
GATE_EXPRESSION_BINS = 10
WEIGHT_CAP_QUANTILE = 0.95
OTSU_BINS = 256  # as skimage.filters.threshold_otsu
MAX_PLACEMENT_PASSES = 5


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC of ``pos`` above ``neg`` (ties count half)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _heldout_auroc(loadings: np.ndarray, members: np.ndarray, background: np.ndarray, rng: np.random.Generator,
                   folds: int) -> float:
    """Mean AUROC over folds: direction from the other folds' members, held-out members vs ``background`` genes."""
    members = members.copy()
    rng.shuffle(members)
    bg = background.copy()
    bg[members] = False
    mean_all = loadings.mean(axis=0)
    aucs = []
    for k in range(folds):
        held = members[k::folds]
        train = np.setdiff1d(members, held)
        if len(held) == 0 or len(train) == 0:
            continue
        score = loadings @ (loadings[train].mean(axis=0) - mean_all)
        aucs.append(_auroc(score[held], score[bg]))
    return float(np.mean(aucs)) if aucs else float("nan")


def structure_gate(loadings: np.ndarray, marker_idx: np.ndarray, non_reference: np.ndarray, gene_totals: np.ndarray,
                   *, seed: int = 0, n_null: int = GATE_NULL_SETS, folds: int = GATE_FOLDS) -> Dict[str, Any]:
    """Held-out marker AUROC on W against expression-matched random gene sets (see module docstring).

    ``marker_idx``: rows of ``loadings`` that are this structure's markers; ``non_reference``: boolean mask of genes in no
    reference set (the background and the pool of null genes); ``gene_totals``: transcript count per row."""
    out: Dict[str, Any] = {"markers": int(len(marker_idx)), "auroc": float("nan"), "null_mean": float("nan"),
                           "null_sd": float("nan"), "z": float("nan"), "pass": False}
    if len(marker_idx) < folds:
        return out
    rng = np.random.default_rng(seed)
    real = _heldout_auroc(loadings, np.asarray(marker_idx), non_reference, rng, folds)
    decile = pd.qcut(np.log1p(np.asarray(gene_totals, float)), GATE_EXPRESSION_BINS, labels=False, duplicates="drop")
    decile = np.asarray(decile)
    pools = {d: np.flatnonzero(non_reference & (decile == d)) for d in np.unique(decile)}
    need = {d: int((decile[marker_idx] == d).sum()) for d in pools}
    null = []
    for _ in range(n_null):
        pick = np.concatenate([rng.choice(pools[d], min(len(pools[d]), need[d]), replace=False) for d in pools])
        null.append(_heldout_auroc(loadings, pick, non_reference, rng, folds))
    null = np.asarray(null, float)
    z = (real - null.mean()) / max(float(null.std()), 1e-6)
    out.update(auroc=real, null_mean=float(null.mean()), null_sd=float(null.std()), z=float(z), **{"pass": bool(np.isfinite(z) and z >= GATE_Z)})
    return out


def gene_weights(loadings: np.ndarray, marker_idx: np.ndarray) -> np.ndarray:
    """Per-gene weight in [0, 1]: W projected on (mean marker W - mean W), divided by the markers' cap quantile."""
    score = loadings @ (loadings[marker_idx].mean(axis=0) - loadings.mean(axis=0))
    cap = float(np.quantile(score[marker_idx], WEIGHT_CAP_QUANTILE))
    if not np.isfinite(cap) or cap <= 0:
        return np.zeros(len(score))
    return np.clip(score / cap, 0.0, 1.0)


def otsu_threshold(values: np.ndarray, nbins: int = OTSU_BINS) -> float:
    """Otsu threshold with the histogram rule of skimage.filters.threshold_otsu (bin centres, between-class variance)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return lo
    hist, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(float)
    w1, w2 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centres) / np.maximum(w1, 1e-300)
    m2 = (np.cumsum((hist * centres)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1e-300))[::-1]
    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centres[:-1][np.argmax(between)])


def _within_group_pct_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Percentile rank (average ties, as pandas rank(pct=True)) of each value within its group."""
    return pd.Series(values).groupby(groups).rank(pct=True).to_numpy()


def place_by_quota(spot_group: np.ndarray, spot_size: np.ndarray, rank_score: Mapping[str, np.ndarray],
                   quota: Mapping[str, np.ndarray]) -> np.ndarray:
    """Assign spots of each group (cell side) to structures until each structure's quota (in transcripts) is filled.

    ``rank_score[s]``: higher = earlier; ``quota[s]``: per group (indexed by ``spot_group`` codes). A spot goes to the
    structure in whose ranking it comes earliest relative to that structure's quota, (before + size / 2) / quota <= 1;
    spots a structure loses to another are skipped and the structure continues down its ranking in the next pass.
    Returns the structure index per spot (-1 = none)."""
    names = list(rank_score)
    n = len(spot_group)
    label = np.full(n, -1)
    left = {s: np.asarray(quota[s], float).copy() for s in names}
    for _ in range(MAX_PLACEMENT_PASSES):
        free = np.flatnonzero(label < 0)
        if not len(free):
            break
        position = np.full((len(names), len(free)), np.inf)
        for k, s in enumerate(names):
            g, size, score = spot_group[free], spot_size[free], rank_score[s][free]
            order = np.lexsort((-score, g))
            before = np.zeros(len(free))
            cum = np.cumsum(size[order])
            starts = np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]
            group_start_cum = np.repeat(np.r_[0.0, cum[starts[1:] - 1]], np.diff(np.r_[starts, len(order)]))
            before[order] = cum - size[order] - group_start_cum
            q = left[s][g]
            pos = np.where(q > 0, (before + size / 2) / np.maximum(q, 1e-12), np.inf)
            position[k] = np.where(pos <= 1, pos, np.inf)
        best = np.argmin(position, axis=0)
        take = np.isfinite(position[best, np.arange(len(free))])
        if not take.any():
            break
        label[free[take]] = best[take]
        for k, s in enumerate(names):
            got = take & (best == k)
            left[s] -= np.bincount(spot_group[free[got]], weights=spot_size[free[got]], minlength=len(left[s]))
    return label


def annotate_lens_count(
    genes: np.ndarray,
    inside: np.ndarray,
    spot_codes: np.ndarray,
    spot_cell: np.ndarray,
    spot_nuclear: np.ndarray,
    structures: Mapping[bool, Sequence[str]],
    marker_sets: Mapping[str, Sequence[str]],
    lens: Mapping[str, Any],
    axis_scores: Mapping[str, np.ndarray],
    broad_label: Mapping[bool, str],
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Per-MetaSpot labels by the LENS counting rule and a JSON-serializable report.

    ``genes`` (upper case), ``inside`` and ``spot_codes`` are per transcript; ``spot_cell`` / ``spot_nuclear`` per MetaSpot
    (codes 0..n_spots-1); ``structures`` maps side (True = nucleus) to its structure names; ``lens`` holds gene_names,
    gene_loadings and background; ``axis_scores[s]`` per MetaSpot (may be all NaN)."""
    raw = lens.get("background", "learned")  # a numpy scalar after an h5ad round trip
    background = str(raw.item() if isinstance(raw, np.ndarray) else raw)
    if background != "own":
        raise ValueError("annotation.evidence_label_rule: lens_count needs LENS trained with module5.lens_background: own "
                         f"(found {background!r}); the learned background also learns cell-type composition")
    vocab = pd.Index(pd.Series(np.asarray(lens["gene_names"]).astype(str)).str.upper())
    loadings = np.asarray(lens["gene_loadings"], dtype=np.float64)
    n_spots = len(spot_cell)
    spot_size = np.bincount(spot_codes, minlength=n_spots).astype(float)
    gene_codes, gene_names = pd.factorize(pd.Series(genes))
    totals = pd.Series(np.bincount(gene_codes), index=gene_names).reindex(vocab).fillna(0).to_numpy()
    all_markers = set().union(*[set(map(str.upper, v)) for v in marker_sets.values()]) if marker_sets else set()
    non_reference = ~vocab.isin(list(all_markers))
    transcript_vocab = vocab.get_indexer(gene_names)[gene_codes]  # -1 = gene not in the LENS vocabulary (weight 0)

    report: Dict[str, Any] = {"background": background, "gate_z": GATE_Z, "structures": {}}
    labels = np.where(spot_nuclear, broad_label[True], broad_label[False]).astype(object)
    for side, names in structures.items():
        side_spots = np.flatnonzero(spot_nuclear == side)
        if not len(side_spots):
            continue
        on_side = inside == side
        evidence, passed = {}, []
        for name in names:
            marker_idx = np.flatnonzero(vocab.isin([g.upper() for g in marker_sets.get(name, [])]))
            entry = structure_gate(loadings, marker_idx, non_reference, totals, seed=seed)
            report["structures"][name] = entry
            if not entry["pass"]:
                continue
            w = gene_weights(loadings, marker_idx)
            tw = np.where(transcript_vocab >= 0, w[np.maximum(transcript_vocab, 0)], 0.0)
            num = np.bincount(spot_codes[on_side], weights=tw[on_side], minlength=n_spots)
            den = np.bincount(spot_codes[on_side], minlength=n_spots)
            ev = np.where(den > 0, num / np.maximum(den, 1), 0.0)
            tau = otsu_threshold(ev[side_spots])
            entry["threshold"] = tau
            if not np.isfinite(tau) or tau <= 0:
                entry["pass"] = False
                continue
            evidence[name] = ev
            passed.append(name)
        if not passed:
            continue
        ratio = np.column_stack([evidence[s][side_spots] / report["structures"][s]["threshold"] for s in passed])
        best = ratio.argmax(axis=1)
        counted = np.where(ratio[np.arange(len(side_spots)), best] >= 1, best, -1)
        group_codes, _ = pd.factorize(spot_cell[side_spots])
        n_groups = int(group_codes.max()) + 1
        size = spot_size[side_spots]
        quota = {s: np.bincount(group_codes[counted == k], weights=size[counted == k], minlength=n_groups)
                 for k, s in enumerate(passed)}
        rank_score = {}
        for s in passed:
            ev_rank = _within_group_pct_rank(evidence[s][side_spots], group_codes)
            axis = np.asarray(axis_scores.get(s, np.full(n_spots, np.nan)), float)[side_spots]
            rank_score[s] = (ev_rank + _within_group_pct_rank(axis, group_codes)) / 2 if np.isfinite(axis).any() else ev_rank
            rank_score[s] = np.nan_to_num(rank_score[s], nan=0.0)
        placed = place_by_quota(group_codes, size, rank_score, quota)
        labels[side_spots] = np.where(placed >= 0, np.asarray(passed, dtype=object)[np.maximum(placed, 0)], labels[side_spots])
        side_total = size.sum()
        for k, s in enumerate(passed):
            report["structures"][s]["counted_share"] = float(quota[s].sum() / max(side_total, 1))
            report["structures"][s]["label_share"] = float(size[placed == k].sum() / max(side_total, 1))
    report["supported_structures"] = [s for s, e in report["structures"].items() if e.get("pass")]
    return labels, report
