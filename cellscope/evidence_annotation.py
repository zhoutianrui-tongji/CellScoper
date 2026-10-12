"""Evidence-based subcellular annotation with built-in held-out validation.

This module contains no pipeline I/O. It answers four questions in order:

1. Does the dataset resolve subcellular localization at all?  (resolution gate)
2. How much marker evidence does each unit carry for each structure, relative
   to what the owning cell's own RNA mixture predicts?           (score)
3. Is that evidence beyond what the same data produce once gene identities
   are shuffled within cells?                                    (decision)
4. Do marker genes the annotation never saw land in the labelled regions?
                                                                 (validation)

A structure is emitted in the primary label only when step 4 supports it in
this dataset. Structures with weaker, unadjusted support appear only in the
exploratory label; every structure keeps its continuous score and q-value. Everything is computed with sparse matrix-vector products,
so memory scales with the number of transcripts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


NUCLEAR_STRUCTURES = ("Nucleolus", "Lamina", "Nuclear Pore")
CYTOPLASMIC_STRUCTURES = ("ER", "OMM")
FINE_STRUCTURES = NUCLEAR_STRUCTURES + CYTOPLASMIC_STRUCTURES
BROAD_LABEL = {"nuclear": "Nucleus", "cytoplasmic": "Cytosol"}
UNIT_LEVELS = ("metaspot", "metadomain")
_REFERENCE_TO_LABEL = {"Nuclear_Pore": "Nuclear Pore"}

DEFAULTS: Dict[str, Any] = {
    "evidence_score": "propagated",        # marker | propagated
    "evidence_unit": "auto",               # auto | metaspot | metadomain
    "evidence_fdr": 0.05,
    "evidence_validation_folds": 5,
    "evidence_null_sets": 10,
    "evidence_min_gene_transcripts": 50,
    "evidence_marker_margin": 0.2,
    "evidence_min_resolution": 0.3,
    "evidence_bootstrap": 2000,
    "evidence_seed": 0,
}

logger = logging.getLogger("cellscope")


def marker_sets_from_reference(
    reference: pd.DataFrame,
    min_margin: float = 0.2,
) -> Dict[str, List[str]]:
    """Assign each reference gene to its top class when it leads by a margin."""
    values = reference.to_numpy(dtype=float)
    if values.shape[1] < 2 or values.shape[0] == 0:
        return {name: [] for name in FINE_STRUCTURES}
    order = np.argsort(-values, axis=1)
    rows = np.arange(values.shape[0])
    top, second = values[rows, order[:, 0]], values[rows, order[:, 1]]
    labels = np.asarray(
        [_REFERENCE_TO_LABEL.get(str(c), str(c)) for c in reference.columns],
        dtype=object,
    )[order[:, 0]]
    specific = (top > 0) & ((top - second) >= float(min_margin))
    genes = reference.index.astype(str).str.upper().to_numpy()
    return {
        name: sorted(set(genes[specific & (labels == name)]))
        for name in FINE_STRUCTURES
    }


def resolution_check(
    cell_ids: Sequence[Any],
    genes: Sequence[Any],
    in_nucleus: Sequence[bool],
    *,
    min_transcripts: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Split-half reproducibility of each gene's nuclear preference.

    The expectation for a gene is built from the nuclear fraction of the cells
    it occurs in, so cell size and cell type do not create a preference. When
    gene identity is unrelated to position (diffusion, misregistered masks),
    the two halves of the cells disagree and the correlation is near zero.
    """
    from scipy.stats import spearmanr

    cell_codes, cells = pd.factorize(np.asarray(cell_ids), sort=False)
    gene_codes, gene_names = pd.factorize(np.asarray(genes), sort=False)
    inside = np.asarray(in_nucleus, dtype=bool)
    n_cells, n_genes = len(cells), len(gene_names)
    cell_total = np.bincount(cell_codes, minlength=n_cells).astype(float)
    cell_nuclear = np.bincount(cell_codes, weights=inside, minlength=n_cells)
    expected_per_transcript = (cell_nuclear / np.maximum(cell_total, 1))[cell_codes]
    half = np.random.default_rng(seed).integers(0, 2, n_cells)[cell_codes]
    bias = []
    counts = []
    for side in (0, 1):
        mask = half == side
        observed = np.bincount(gene_codes[mask], weights=inside[mask], minlength=n_genes)
        expected = np.bincount(
            gene_codes[mask], weights=expected_per_transcript[mask], minlength=n_genes
        )
        bias.append(np.log2((observed + 0.5) / (expected + 0.5)))
        counts.append(np.bincount(gene_codes[mask], minlength=n_genes))
    usable = (counts[0] >= min_transcripts / 2) & (counts[1] >= min_transcripts / 2)
    result: Dict[str, Any] = {
        "genes_tested": int(usable.sum()),
        "nuclear_fraction": float(inside.mean()) if inside.size else float("nan"),
        "split_half_spearman": float("nan"),
    }
    if usable.sum() >= 30 and 0.0 < result["nuclear_fraction"] < 1.0:
        result["split_half_spearman"] = float(
            spearmanr(bias[0][usable], bias[1][usable])[0]
        )
    return result


class _Compartment:
    """Sparse unit x gene counts with a within-cell mixing expectation."""

    def __init__(
        self,
        counts: sp.csr_matrix,
        cell_codes: np.ndarray,
        domain_codes: np.ndarray,
        genes: np.ndarray,
    ) -> None:
        self.X = counts.tocsr().astype(np.float64)
        self.genes = np.asarray(genes, dtype=object)
        self.cell_codes = np.asarray(cell_codes, dtype=np.int64)
        self.domain_codes = np.asarray(domain_codes, dtype=np.int64)
        self.n_units = self.X.shape[0]
        self.n_cells = int(self.cell_codes.max()) + 1 if self.n_units else 0
        self.n_domains = int(self.domain_codes.max()) + 1 if self.n_units else 0
        self._cells = sp.csr_matrix(
            (np.ones(self.n_units), (self.cell_codes, np.arange(self.n_units))),
            shape=(self.n_cells, self.n_units),
        )
        self._prepare()

    def _prepare(self) -> None:
        self.n = np.asarray(self.X.sum(axis=1)).ravel()
        self.cell_gene = (self._cells @ self.X).tocsr()
        self.cell_total = np.maximum(np.asarray(self.cell_gene.sum(axis=1)).ravel(), 1.0)

    def permuted(self, rng: np.random.Generator) -> "_Compartment":
        """Shuffle gene identities among the transcripts of each cell.

        Unit sizes and every cell's gene totals are preserved, so the mixing
        expectation is unchanged while all within-cell spatial structure is
        removed. Scoring this copy with the same markers gives the null,
        including the optimism of weights learned from the same data.
        """
        coo = self.X.tocoo()
        repeats = coo.data.astype(np.int64)
        units = np.repeat(coo.row, repeats)
        genes = np.repeat(coo.col, repeats)
        cells = self.cell_codes[units]
        by_cell = np.argsort(cells, kind="stable")
        shuffled_within_cell = np.lexsort((rng.random(units.size), cells))
        new_genes = np.empty_like(genes)
        new_genes[shuffled_within_cell] = genes[by_cell]
        counts = sp.csr_matrix(
            (np.ones(units.size), (units, new_genes)), shape=self.X.shape
        )
        return _Compartment(counts, self.cell_codes, self.domain_codes, self.genes)

    def without_genes(self, drop: np.ndarray) -> "_Compartment":
        keep = ~np.asarray(drop, dtype=bool)
        return _Compartment(
            self.X[:, keep], self.cell_codes, self.domain_codes, self.genes[keep]
        )

    def excess(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted observed-minus-expected and its Poisson variance per unit."""
        w = np.asarray(weights, dtype=np.float64)
        expected = self.n * ((self.cell_gene @ w) / self.cell_total)[self.cell_codes]
        variance = self.n * ((self.cell_gene @ (w * w)) / self.cell_total)[self.cell_codes]
        return self.X @ w - expected, variance

    def propagate(self, seed: np.ndarray) -> np.ndarray:
        """Weight every gene by its within-cell co-localization with the seed.

        Non-seed genes are compared against a mixture that excludes the seed
        transcripts, so a seed-rich unit does not mechanically look depleted
        of everything else. Seed genes are scored leave-one-gene-out.
        """
        seed = np.asarray(seed, dtype=bool)
        numerator, _ = self.excess(seed.astype(float))
        weights = np.zeros(self.genes.size, dtype=np.float64)

        rest = ~seed
        if rest.any():
            X_rest = self.X[:, rest]
            n_rest = np.asarray(X_rest.sum(axis=1)).ravel()
            cell_gene = self.cell_gene[:, rest]
            cell_total = np.maximum(np.asarray(cell_gene.sum(axis=1)).ravel(), 1.0)
            pulled = self._cells @ (n_rest * numerator)
            spread = self._cells @ (n_rest * numerator**2)
            covariance = X_rest.T @ numerator - cell_gene.T @ (pulled / cell_total)
            variance = cell_gene.T @ (spread / cell_total)
            weights[rest] = covariance / np.sqrt(np.maximum(variance, 1e-12))

        seed_index = np.flatnonzero(seed)
        for start in range(0, seed_index.size, 64):
            block = seed_index[start:start + 64]
            share = (
                self.cell_gene[:, block].toarray() / self.cell_total[:, None]
            )[self.cell_codes]
            expected = self.n[:, None] * share
            own = self.X[:, block].toarray() - expected
            others = numerator[:, None] - own
            variance = (expected * others**2).sum(axis=0)
            weights[block] = (own * others).sum(axis=0) / np.sqrt(
                np.maximum(variance, 1e-12)
            )
        return weights

    def unit_scores(self, weights: np.ndarray, level: str) -> np.ndarray:
        """z-like evidence per unit; MetaDomain level pools units across cells."""
        numerator, variance = self.excess(weights)
        if level == "metadomain":
            numerator = np.bincount(
                self.domain_codes, weights=numerator, minlength=self.n_domains
            )[self.domain_codes]
            variance = np.bincount(
                self.domain_codes, weights=variance, minlength=self.n_domains
            )[self.domain_codes]
        return numerator / np.sqrt(np.maximum(variance, 1e-12))

    def gene_weights(self, seed: np.ndarray, mode: str) -> np.ndarray:
        if mode == "marker":
            return np.asarray(seed, dtype=np.float64)
        return self.propagate(seed)


def _bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(ranked, 1.0)
    return q


def _tail_p_values(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Upper-tail p-values from a Student-t fitted to the pooled null.

    Calls live far in the tail, where counting null exceedances is noisy
    for any affordable number of replicates. Matching the null's mean,
    variance, and kurtosis gives a smooth tail that is stable across seeds
    while keeping the heavier-than-normal tail small counts produce.
    """
    from scipy.stats import norm, t as student_t

    centre, spread = float(np.mean(null)), float(np.std(null))
    if not np.isfinite(spread) or spread <= 0:
        return np.ones(observed.shape)
    standardized = (null - centre) / spread
    excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    if excess_kurtosis <= 0.05:
        return norm.sf((observed - centre) / spread)
    dof = max(4.0 + 6.0 / excess_kurtosis, 4.5)
    scale = spread * np.sqrt((dof - 2.0) / dof)
    return student_t.sf((observed - centre) / scale, dof)


def _score_and_test(
    data: _Compartment,
    seed: np.ndarray,
    *,
    mode: str,
    levels: Sequence[str],
    null_sets: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Unit scores and BH q-values against a within-cell permutation null.

    Each null replicate reruns the whole scoring, gene weighting included, on
    a copy whose gene identities were shuffled within cells. Whether a called
    program really is the named structure is a separate question, answered by
    the held-out marker validation.
    """
    seed = np.asarray(seed, dtype=bool)
    weights = data.gene_weights(seed, mode)
    scores = {level: data.unit_scores(weights, level) for level in levels}
    # One value per independent test: a MetaDomain is a single test.
    first = {
        level: (
            np.unique(data.domain_codes, return_index=True)[1]
            if level == "metadomain"
            else np.arange(data.n_units)
        )
        for level in levels
    }
    if not seed.any():
        return {level: (scores[level], np.ones(data.n_units)) for level in levels}
    null: Dict[str, List[np.ndarray]] = {level: [] for level in levels}
    for _ in range(int(null_sets)):
        shuffled = data.permuted(rng)
        null_weights = shuffled.gene_weights(seed, mode)
        for level in levels:
            null[level].append(shuffled.unit_scores(null_weights, level)[first[level]])
    result = {}
    for level in levels:
        observed = scores[level][first[level]]
        q_first = _bh_qvalues(_tail_p_values(observed, np.concatenate(null[level])))
        if level == "metadomain":
            lookup = np.ones(data.n_domains)
            lookup[data.domain_codes[first[level]]] = q_first
            q = lookup[data.domain_codes]
        else:
            q = q_first
        result[level] = (scores[level], q)
    return result


def _marker_gene_enrichment(
    observed: np.ndarray,
    expected: np.ndarray,
    is_marker: np.ndarray,
    is_background: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
    support_alpha: float,
    min_expected: float = 1.0,
) -> Dict[str, float]:
    """How much more enriched is the typical marker gene than the typical gene?

    Each gene contributes one observed/expected ratio for the labelled
    regions. Comparing medians, with genes as the resampling unit, means the
    answer cannot be carried by one abundant gene, and a label that merely
    excludes some other strongly localized program gains nothing: generic
    genes are enriched there exactly as much as the markers are.
    """
    usable = expected >= float(min_expected)
    ratio = np.log2((observed + 0.5) / (expected + 0.5))
    marker = ratio[usable & is_marker]
    background = ratio[usable & is_background]
    result = {
        "marker_enrichment": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "support_bound": float("nan"),
        "marker_genes_scored": int(marker.size),
        "background_genes_scored": int(background.size),
    }
    if marker.size < 4 or background.size < 20:
        return result
    difference = float(np.median(marker) - np.median(background))
    boot = np.median(
        marker[rng.integers(0, marker.size, (int(n_boot), marker.size))], axis=1
    ) - np.median(
        background[rng.integers(0, background.size, (int(n_boot), background.size))],
        axis=1,
    )
    result.update(
        marker_enrichment=float(2.0**difference),
        ci_low=float(2.0 ** np.percentile(boot, 2.5)),
        ci_high=float(2.0 ** np.percentile(boot, 97.5)),
        support_bound=float(2.0 ** np.percentile(boot, 100.0 * support_alpha)),
    )
    return result


def _validate_structure(
    data: _Compartment,
    seed: np.ndarray,
    eligible: np.ndarray,
    *,
    mode: str,
    levels: Sequence[str],
    fdr: float,
    folds: int,
    null_sets: int,
    n_boot: int,
    rng: np.random.Generator,
    support_alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Held-out marker enrichment of the labels each unit level would emit.

    Every eligible gene is held out exactly once. In its fold it is removed
    from the data before scoring, so neither its transcripts nor its
    co-localization can influence the labels it is then scored against. The
    expectation for a held-out gene is the share of each cell's remaining
    transcripts that fall in labelled units.
    """
    seed = np.asarray(seed, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    fold_of = np.full(seed.size, -1)
    for group in (np.flatnonzero(seed), np.flatnonzero(eligible & ~seed)):
        fold_of[rng.permutation(group)] = np.arange(group.size) % folds
    observed = {level: np.zeros(seed.size) for level in levels}
    expected = {level: np.zeros(seed.size) for level in levels}
    units = {level: 0 for level in levels}
    for fold in range(folds):
        held = fold_of == fold
        if not (held & seed).any() or not (seed & ~held).any():
            continue
        train = data.without_genes(held)
        tested = _score_and_test(
            train,
            seed[~held],
            mode=mode,
            levels=levels,
            null_sets=null_sets,
            rng=rng,
        )
        held_counts = data.X[:, held]
        held_cell_counts = data.cell_gene[:, held]
        cell_size = np.maximum(
            np.bincount(data.cell_codes, weights=train.n, minlength=data.n_cells), 1.0
        )
        for level in levels:
            called = (tested[level][1] <= fdr).astype(np.float64)
            labelled_share = (
                np.bincount(
                    data.cell_codes, weights=train.n * called, minlength=data.n_cells
                )
                / cell_size
            )
            observed[level][held] = held_counts.T @ called
            expected[level][held] = held_cell_counts.T @ labelled_share
            units[level] += int(called.sum())
    out = {}
    for level in levels:
        result = _marker_gene_enrichment(
            observed[level],
            expected[level],
            seed,
            eligible & ~seed,
            n_boot=n_boot,
            rng=rng,
            support_alpha=support_alpha,
        )
        result.update(
            labelled_units_per_fold=float(units[level] / max(folds, 1)),
            supported=bool(
                np.isfinite(result["support_bound"]) and result["support_bound"] > 1.0
            ),
        )
        out[level] = result
    return out


def annotate_compartment(
    data: _Compartment,
    marker_sets: Mapping[str, Sequence[str]],
    structures: Sequence[str],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Score, test, and validate every structure of one broad compartment."""
    mode = str(config["evidence_score"]).lower()
    unit = str(config["evidence_unit"]).lower()
    if mode not in {"marker", "propagated"}:
        raise ValueError("annotation.evidence_score must be 'marker' or 'propagated'")
    if unit not in {"auto", *UNIT_LEVELS}:
        raise ValueError(
            "annotation.evidence_unit must be 'auto', 'metaspot', or 'metadomain'"
        )
    levels = UNIT_LEVELS if unit == "auto" else (unit,)
    gene_total = np.asarray(data.X.sum(axis=0)).ravel()
    eligible = gene_total >= int(config["evidence_min_gene_transcripts"])
    columns: Dict[str, np.ndarray] = {}
    report: Dict[str, Any] = {}
    for name in structures:
        seed = np.isin(data.genes, list(marker_sets.get(name, ()))) & eligible
        entry: Dict[str, Any] = {"marker_genes_used": int(seed.sum()), "levels": {}}
        best_level, best_bound = None, 1.0
        if seed.sum() >= max(4, int(config["evidence_validation_folds"])):
            entry["levels"] = _validate_structure(
                data,
                seed,
                eligible,
                mode=mode,
                levels=levels,
                fdr=float(config["evidence_fdr"]),
                folds=int(config["evidence_validation_folds"]),
                null_sets=int(config["evidence_null_sets"]),
                n_boot=int(config["evidence_bootstrap"]),
                rng=rng,
                # Ten structure/level combinations are tried per dataset.
                support_alpha=0.05 / (len(FINE_STRUCTURES) * len(UNIT_LEVELS)),
            )
            for level, validation in entry["levels"].items():
                if validation["supported"] and validation["support_bound"] > best_bound:
                    best_level, best_bound = level, validation["support_bound"]
        else:
            entry["skipped"] = "too few marker genes measured in this compartment"
        entry["supported"] = best_level is not None
        # Suggestive: the plain 95% interval excludes 1 but the bound adjusted
        # for all structure/level combinations does not. Such structures are
        # offered as exploratory labels; anything weaker is not labelled.
        entry["suggestive"] = bool(
            entry["supported"]
            or any(
                np.isfinite(v["ci_low"]) and v["ci_low"] > 1.0
                for v in entry["levels"].values()
            )
        )
        entry["level"] = best_level or max(
            entry["levels"],
            key=lambda lv: np.nan_to_num(
                entry["levels"][lv]["marker_enrichment"], nan=0.0
            ),
            default=levels[0],
        )
        scores, q = _score_and_test(
            data,
            seed,
            mode=mode,
            levels=(entry["level"],),
            null_sets=int(config["evidence_null_sets"]),
            rng=rng,
        )[entry["level"]]
        entry["units_called"] = int((q <= float(config["evidence_fdr"])).sum())
        columns[f"score__{name}"] = scores
        columns[f"q__{name}"] = q
        report[name] = entry
    # Two names for one program: how much do the called units coincide?
    fdr = float(config["evidence_fdr"])
    for name in structures:
        mine = columns[f"q__{name}"] <= fdr
        report[name]["called_overlap"] = {
            other: float(
                (mine & (columns[f"q__{other}"] <= fdr)).sum()
                / max((mine | (columns[f"q__{other}"] <= fdr)).sum(), 1)
            )
            for other in structures
            if other != name
        }
    return pd.DataFrame(columns), report


def _choose_labels(
    table: pd.DataFrame,
    structures: Sequence[str],
    allowed: Sequence[str],
    broad: str,
    fdr: float,
) -> np.ndarray:
    """Most significant allowed structure per unit, else the broad label."""
    labels = np.full(len(table), broad, dtype=object)
    best_q = np.full(len(table), np.inf)
    best_score = np.full(len(table), -np.inf)
    for name in structures:
        if name not in allowed:
            continue
        q = table[f"q__{name}"].to_numpy()
        score = table[f"score__{name}"].to_numpy()
        better = (q <= fdr) & ((q < best_q) | ((q == best_q) & (score > best_score)))
        labels[better] = name
        best_q[better], best_score[better] = q[better], score[better]
    return labels


# A supported structure also takes every unit scoring at least this many
# within-cell standard deviations above the cell's mean. On held-out markers 1.0
# roughly doubles the labelled area for a small loss of enrichment; 0.5 loses
# clearly more.
_EXTEND_Z = 1.0


def _seat_structures(
    scores: np.ndarray, cells: np.ndarray, extend: Sequence[int] = ()
) -> np.ndarray:
    """Give every structure its best-scoring unit in every cell.

    ``scores`` is units x structures. Returns the structure index per unit, or
    -1. Cells with too few units to seat every structure and keep one unit for
    the broad label are left unseated. Structures listed in ``extend`` (the
    validated ones) additionally take the other units that score clearly above
    the cell's average, so they can occupy several places in a cell.
    """
    seats = np.full(len(scores), -1)
    scored = np.flatnonzero(np.isfinite(scores).all(axis=0))  # structures without markers have no score
    wide = np.flatnonzero(np.isin(scored, list(extend)))
    order = np.argsort(cells, kind="stable")
    bounds = np.flatnonzero(np.diff(cells[order])) + 1
    for rows in np.split(order, bounds):
        if len(rows) <= len(scored):
            continue
        local = scores[np.ix_(rows, scored)]
        local = (local - local.mean(axis=0)) / (local.std(axis=0) + 1e-9)
        free = np.ones(len(rows), dtype=bool)
        # The structure with the clearest best unit chooses first.
        for j in np.argsort(-local.max(axis=0)):
            pick = np.flatnonzero(free)[np.argmax(local[free, j])]
            seats[rows[pick]] = scored[j]
            free[pick] = False
        if wide.size:
            best = wide[np.argmax(local[:, wide], axis=1)]
            more = free & (local[np.arange(len(rows)), best] >= _EXTEND_Z)
            seats[rows[more]] = scored[best[more]]
    return seats


def run_evidence_annotation(
    transcripts: pd.DataFrame,
    reference: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Annotate MetaSpots from a transcript table.

    ``transcripts`` needs ``cell_id``, ``feature_name``, ``metaspot``,
    ``meta-domain`` and ``in_nucleus``. Returns one row per MetaSpot with the
    final, validated and exploratory labels, the support level of each final
    label, per-structure scores and q-values,
    plus a JSON-serializable report. The final labels place all structures in
    every cell and are not gated by validation; the report lists which
    structures held-out markers support.
    """
    options = dict(DEFAULTS)
    options.update({k: v for k, v in (config or {}).items() if k in DEFAULTS and v is not None})
    required = {"cell_id", "feature_name", "metaspot", "meta-domain", "in_nucleus"}
    missing = sorted(required - set(transcripts.columns))
    if missing:
        raise ValueError(f"evidence annotation is missing transcript columns: {missing}")
    rng = np.random.default_rng(int(options["evidence_seed"]))
    genes = transcripts["feature_name"].astype(str).str.upper().to_numpy()
    inside = transcripts["in_nucleus"].to_numpy(dtype=bool)
    spot_codes, spots = pd.factorize(transcripts["metaspot"].to_numpy(), sort=False)
    gene_codes, gene_names = pd.factorize(genes, sort=False)
    cell_codes, _ = pd.factorize(transcripts["cell_id"].to_numpy(), sort=False)
    domain_codes, _ = pd.factorize(transcripts["meta-domain"].astype(str).to_numpy(), sort=False)
    n_spots = len(spots)
    counts = sp.csr_matrix(
        (np.ones(len(transcripts)), (spot_codes, gene_codes)),
        shape=(n_spots, len(gene_names)),
    )
    first_row = np.full(n_spots, -1)
    first_row[spot_codes[::-1]] = np.arange(len(transcripts))[::-1]
    spot_cell, spot_domain = cell_codes[first_row], domain_codes[first_row]
    nuclear_share = np.bincount(spot_codes, weights=inside, minlength=n_spots) / np.maximum(
        np.bincount(spot_codes, minlength=n_spots), 1
    )
    spot_nuclear = nuclear_share > 0.5

    resolution = resolution_check(
        transcripts["cell_id"].to_numpy(), genes, inside, seed=int(options["evidence_seed"])
    )
    rho = resolution["split_half_spearman"]
    resolution["passed"] = bool(
        not np.isfinite(rho) or rho >= float(options["evidence_min_resolution"])
    )
    resolution["determinable"] = bool(np.isfinite(rho))
    marker_sets = marker_sets_from_reference(
        reference, float(options["evidence_marker_margin"])
    )
    report: Dict[str, Any] = {
        "method": "evidence",
        "settings": {k: options[k] for k in DEFAULTS},
        "resolution": resolution,
        "reference_marker_genes": {k: len(v) for k, v in marker_sets.items()},
        "structures": {},
    }
    result = pd.DataFrame({"metaspot": spots})
    result["subcellular_broad_compartment"] = np.where(
        spot_nuclear, BROAD_LABEL["nuclear"], BROAD_LABEL["cytoplasmic"]
    )
    primary = result["subcellular_broad_compartment"].to_numpy(dtype=object).copy()
    exploratory = primary.copy()
    complete = primary.copy()
    for name in FINE_STRUCTURES:
        result[f"subcellular_score__{name}"] = np.nan
        result[f"subcellular_q__{name}"] = np.nan

    if not resolution["passed"]:
        logger.warning(
            "Evidence annotation: gene nuclear preference is not reproducible "
            "between halves of the cells (Spearman %.2f < %.2f). The data do "
            "not resolve subcellular localization; only Nucleus/Cytosol is "
            "reported.",
            rho,
            float(options["evidence_min_resolution"]),
        )
    else:
        for mask, structures, broad in (
            (spot_nuclear, NUCLEAR_STRUCTURES, BROAD_LABEL["nuclear"]),
            (~spot_nuclear, CYTOPLASMIC_STRUCTURES, BROAD_LABEL["cytoplasmic"]),
        ):
            rows = np.flatnonzero(mask)
            if rows.size < 10:
                continue
            sub_counts = counts[rows]
            present = np.asarray(sub_counts.sum(axis=0)).ravel() > 0
            _, local_cells = np.unique(spot_cell[rows], return_inverse=True)
            _, local_domains = np.unique(spot_domain[rows], return_inverse=True)
            data = _Compartment(
                sub_counts[:, present], local_cells, local_domains, gene_names[present]
            )
            table, structure_report = annotate_compartment(
                data, marker_sets, structures, options, rng
            )
            report["structures"].update(structure_report)
            supported = [s for s in structures if structure_report[s]["supported"]]
            suggestive = [s for s in structures if structure_report[s]["suggestive"]]
            fdr = float(options["evidence_fdr"])
            primary[rows] = _choose_labels(table, structures, supported, broad, fdr)
            exploratory[rows] = _choose_labels(table, structures, suggestive, broad, fdr)
            # Final view: every cell shows every structure at its most likely
            # unit; validated structures also take their other high-scoring units.
            scores = table[[f"score__{name}" for name in structures]].to_numpy(dtype=float)
            seats = _seat_structures(
                scores, local_cells, [structures.index(s) for s in supported]
            )
            seated = np.where(seats >= 0, np.asarray(structures, dtype=object)[seats], primary[rows])
            complete[rows] = seated
            for name in structures:
                result.loc[rows, f"subcellular_score__{name}"] = table[f"score__{name}"].to_numpy()
                result.loc[rows, f"subcellular_q__{name}"] = table[f"q__{name}"].to_numpy()
    # The final label shows all seven structures in every cell; the validated
    # view keeps only what held-out markers support in this dataset.
    result["subcellular_annotation"] = complete
    result["subcellular_annotation_validated"] = primary
    result["subcellular_annotation_exploratory"] = exploratory
    report["supported_structures"] = [
        name for name, entry in report["structures"].items() if entry.get("supported")
    ]
    report["unsupported_structures"] = [
        name for name in FINE_STRUCTURES if name not in report["supported_structures"]
    ]
    # How far each final label can be trusted, so downstream code does not have
    # to read the report: broad labels come from the nuclear segmentation, fine
    # labels are either validated on held-out markers or placed by reference only.
    result["subcellular_annotation_support"] = np.where(
        np.isin(complete, list(BROAD_LABEL.values())),
        "segmentation",
        np.where(
            np.isin(complete, report["supported_structures"]), "validated", "unvalidated"
        ),
    )
    report["label_metaspot_fraction"] = {
        str(k): float(v)
        for k, v in pd.Series(complete).value_counts(normalize=True).items()
    }
    report["validated_label_metaspot_fraction"] = {
        str(k): float(v)
        for k, v in pd.Series(primary).value_counts(normalize=True).items()
    }
    return result, report
