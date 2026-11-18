"""Subcellular annotation with built-in held-out validation (structure-axis rule).

This module contains no pipeline I/O. It answers three questions in order:

1. Does the dataset resolve subcellular localization at all? Split-half reproducibility of each gene's nuclear
   preference (resolution gate); when it fails only Nucleus / Cytosol is reported.
2. Where is each structure? A continuous score per MetaSpot read from the representations along the direction of the
   structure's reference markers, labelled above a threshold set on gene-shuffled cells at a target false-label share
   (``structure_axes.annotate``).
3. Do marker genes the annotation never saw land in the labelled regions? (held-out validation, reported; the
   ``validated`` view keeps only supported structures)

Until 2026-11-03 a count-based rule with per-cell seats and MetaDomain borrowing (``evidence_label_rule: legacy``) was
also available; on gene-shuffled input it gave 3.8-10.5% false labels against 0-0.9% for the axis rule.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .runtime import _preferred_n_jobs


NUCLEAR_STRUCTURES = ("Nucleolus", "Lamina", "Nuclear Pore", "Nuclear Speckle")
CYTOPLASMIC_STRUCTURES = ("ER", "OMM")
FINE_STRUCTURES = NUCLEAR_STRUCTURES + CYTOPLASMIC_STRUCTURES
BROAD_LABEL = {"nuclear": "Nucleus", "cytoplasmic": "Cytosol"}
_REFERENCE_TO_LABEL = {"Nuclear_Pore": "Nuclear Pore", "Nuclear_Speckle": "Nuclear Speckle"}
# Mitochondrially encoded RNAs, when the data measure at least this many, replace the OMM reference markers.
_MIN_MITO_ENCODED = 4

DEFAULTS: Dict[str, Any] = {
    "evidence_validation_folds": 5,
    "evidence_max_cells": 10000,
    "evidence_min_gene_transcripts": 50,
    "evidence_marker_margin": 0.2,
    "evidence_min_resolution": 0.3,
    "evidence_bootstrap": 2000,
    "evidence_seed": 0,
    "evidence_axis_source": "auto",  # auto (ge+lens when the model ran, else ge) | ge | lens | ge+lens
    "evidence_axis_q": 0.2,  # target share of false labels per structure
    "evidence_axis_validation_cells": 3000,  # cells of the held-out report
    "evidence_axis_smooth_transcripts": 50,  # scores are averaged over about this many transcripts
    "evidence_axis_radial_bands": 0,  # >0: cytoplasm scores centred within this many bands of distance to the cell edge (0 = off)
    "evidence_label_rule": "lens_count",  # lens_count (default since v1.0.0; amount by LENS counting, see lens_count.py) | axis
}
LABEL_RULES = ("axis", "lens_count")

logger = logging.getLogger("cellscoper")


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


def _median_enrichment(
    observed: np.ndarray, expected: np.ndarray, genes: np.ndarray, background: np.ndarray, min_expected: float = 1.0
) -> float:
    """Point estimate of _marker_gene_enrichment for one gene set (nan when too few genes)."""
    usable = expected >= min_expected
    ratio = np.log2((observed + 0.5) / (expected + 0.5))
    a, b = ratio[usable & genes], ratio[usable & background]
    if a.size < 4 or b.size < 20:
        return float("nan")
    return float(2.0 ** (np.median(a) - np.median(b)))


def _seat_structures(scores: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Give every structure its best-scoring unit in every cell (display only).

    ``scores`` is units x structures. Returns the structure index per unit, or -1. Cells with too few units to seat
    every structure and keep one unit for the broad label are left unseated.
    """
    seats = np.full(len(scores), -1)
    scored = np.flatnonzero(np.isfinite(scores).all(axis=0))  # structures without markers have no score
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
    return seats


def run_evidence_annotation(
    transcripts: pd.DataFrame,
    reference: pd.DataFrame,
    config: Optional[Mapping[str, Any]] = None,
    axis_inputs: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Annotate MetaSpots from a transcript table with the structure-axis rule (``structure_axes.annotate``).

    ``transcripts`` needs ``cell_id``, ``feature_name``, ``metaspot`` and ``in_nucleus``; ``axis_inputs`` holds
    spot_names, ge + ge_block and/or lens + z. Returns one row per MetaSpot with the final, validated and exploratory
    labels, the support level of each final label, per-structure scores and false-label estimates, plus a
    JSON-serializable report. ``subcellular_annotation_placed`` keeps the per-cell best unit of every structure for
    display only.
    """
    options = dict(DEFAULTS)
    options.update({k: v for k, v in (config or {}).items() if k in DEFAULTS and v is not None})
    if options["evidence_label_rule"] not in LABEL_RULES:
        raise ValueError(f"annotation.evidence_label_rule must be one of {LABEL_RULES}, got {options['evidence_label_rule']!r}")
    required = {"cell_id", "feature_name", "metaspot", "in_nucleus"}
    missing = sorted(required - set(transcripts.columns))
    if missing:
        raise ValueError(f"evidence annotation is missing transcript columns: {missing}")
    gene_col = transcripts["feature_name"]
    if isinstance(gene_col.dtype, pd.CategoricalDtype):  # 只转换类别一次：几亿行共用同一批字符串对象
        genes = gene_col.cat.categories.astype(str).str.upper().to_numpy(dtype=object)[gene_col.cat.codes.to_numpy()]
    else:
        genes = gene_col.astype(str).str.upper().to_numpy()
    inside = transcripts["in_nucleus"].to_numpy(dtype=bool)
    spot_codes, spots = pd.factorize(transcripts["metaspot"].to_numpy(), sort=False)
    cell_codes, _ = pd.factorize(transcripts["cell_id"].to_numpy(), sort=False)
    n_spots = len(spots)
    first_row = np.full(n_spots, -1)
    first_row[spot_codes[::-1]] = np.arange(len(transcripts))[::-1]
    spot_cell = cell_codes[first_row]
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
    # Mitochondrially encoded RNAs exist only inside mitochondria. When the data
    # measure them (polyA capture, custom probes) they alone mark the mitochondrion
    # (the OMM slot); nuclear-encoded OMM lists are a weak proxy (Atera BRCA: no
    # better than expression-matched random genes).
    mito_encoded = sorted({g for g in np.unique(genes) if g.startswith("MT-")})
    if len(mito_encoded) >= _MIN_MITO_ENCODED:
        marker_sets["OMM"] = mito_encoded
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
        from .structure_axes import annotate as annotate_axes

        table, structure_report = annotate_axes(
            transcripts,
            axis_inputs or {},
            {True: NUCLEAR_STRUCTURES, False: CYTOPLASMIC_STRUCTURES},
            {True: BROAD_LABEL["nuclear"], False: BROAD_LABEL["cytoplasmic"]},
            marker_sets,
            options,
            n_jobs=_preferred_n_jobs(),
        )
        rows = pd.Index(table["metaspot"].astype(str)).get_indexer(pd.Index(np.asarray(spots)).astype(str))
        report["structures"].update(structure_report)
        complete[:] = table["label"].to_numpy(dtype=object)[rows]
        placed = complete.copy()
        fine = {True: NUCLEAR_STRUCTURES, False: CYTOPLASMIC_STRUCTURES}
        for side, mask in ((True, spot_nuclear), (False, ~spot_nuclear)):
            allowed = [n for n in fine[side] if structure_report.get(n, {}).get("supported")]
            hints = [n for n in fine[side] if structure_report.get(n, {}).get("suggestive")]
            primary[mask] = np.where(np.isin(complete[mask], allowed), complete[mask], primary[mask])
            exploratory[mask] = np.where(np.isin(complete[mask], hints), complete[mask], exploratory[mask])
            names = [n for n in fine[side] if f"score__{n}" in table and table[f"score__{n}"].notna().any()]
            if names and mask.any():
                _, local_cells = np.unique(spot_cell[mask], return_inverse=True)
                scores = np.column_stack([table[f"score__{n}"].to_numpy()[rows][mask] for n in names])
                seats = _seat_structures(scores, local_cells)
                placed[mask] = np.where(seats >= 0, np.asarray(names, dtype=object)[np.maximum(seats, 0)], placed[mask])
        result["subcellular_annotation_placed"] = placed
        for name in FINE_STRUCTURES:
            if f"score__{name}" in table:
                result[f"subcellular_score__{name}"] = table[f"score__{name}"].to_numpy()[rows]
                result[f"subcellular_q__{name}"] = table[f"q__{name}"].to_numpy()[rows]
        if options["evidence_label_rule"] == "lens_count":
            # 量由 LENS 计数决定、位置按计数证据与结构轴分数的排名平均（lens_count.py）；替换最终、验证、探索三种标签
            from .lens_count import annotate_lens_count

            lens = (axis_inputs or {}).get("lens")
            if lens is None or "gene_loadings" not in lens:
                raise ValueError("annotation.evidence_label_rule: lens_count needs the LENS model (adata1.uns['lens'])")
            new_labels, lens_report = annotate_lens_count(
                genes, inside, spot_codes, spot_cell, spot_nuclear,
                {True: NUCLEAR_STRUCTURES, False: CYTOPLASMIC_STRUCTURES}, marker_sets, lens,
                {name: result[f"subcellular_score__{name}"].to_numpy() for name in FINE_STRUCTURES},
                {True: BROAD_LABEL["nuclear"], False: BROAD_LABEL["cytoplasmic"]}, seed=int(options["evidence_seed"]),
            )
            complete[:] = new_labels
            primary[:] = new_labels
            exploratory[:] = new_labels
            report["lens_count"] = lens_report
    # The final label shows every structure in every cell; the validated
    # view keeps only what held-out markers support in this dataset.
    result["subcellular_annotation"] = complete
    result["subcellular_annotation_validated"] = primary
    result["subcellular_annotation_exploratory"] = exploratory
    report["label_rule"] = options["evidence_label_rule"]
    if "lens_count" in report:  # 计数标签只给过 LENS 门槛的结构
        report["supported_structures"] = list(report["lens_count"]["supported_structures"])
    else:
        report["supported_structures"] = [
            name
            for name, entry in report["structures"].items()
            if entry.get("supported") or entry.get("rank_supported")
        ]
    report["unsupported_structures"] = [
        name
        for name in FINE_STRUCTURES
        if name not in report["supported_structures"]
        and report["structures"].get(name, {}).get("marker_genes_used")
    ]
    report["structures_without_markers"] = [
        name
        for name in FINE_STRUCTURES
        if not report["structures"].get(name, {}).get("marker_genes_used")
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
