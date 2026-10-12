#!/usr/bin/env python3
"""Evaluate a CellScoper run without reusing the genes that produced its labels.

Subcommands
-----------
splits    Split reference marker genes into annotation/held-out halves.
evaluate  Score one run directory (broad accuracy, label composition, and,
          when a held-out table is given, held-out marker enrichment).
compare   Agreement between two runs of the same input.

Held-out marker enrichment
--------------------------
A run is annotated with ``annotation.reference_exclude_genes_file`` pointing at
the held-out genes, so those genes cannot influence any label. For each fine
label L the score asks: within the same broad compartment and the same cell,
are transcripts of held-out L markers over-represented in regions labelled L,
relative to transcripts of other genes that the annotation also never saw?
The statistic is a Mantel-Haenszel odds ratio stratified by cell, so it is not
inflated by cell types that both express more L markers and receive more L
labels. Confidence intervals come from a cell-level bootstrap.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


NUCLEAR_LABELS = ("Nucleus", "Nucleolus", "Lamina", "Nuclear Pore", "Nuclear Speckle")
CYTOPLASMIC_LABELS = ("Cytosol", "ER", "OMM")
FINE_LABELS = ("Nucleolus", "Lamina", "Nuclear Pore", "Nuclear Speckle", "ER", "OMM")
ALL_LABELS = NUCLEAR_LABELS + CYTOPLASMIC_LABELS
_GOLD_TO_LABEL = {"Nuclear_Pore": "Nuclear Pore", "Nuclear_Speckle": "Nuclear Speckle"}


def _final_table(run_dir: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    parquet = os.path.join(run_dir, "final_data.parquet")
    if os.path.isfile(parquet):
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(parquet).schema_arrow.names)
        use = [c for c in columns if c in available] if columns else None
        return pd.read_parquet(parquet, columns=use)
    csv = os.path.join(run_dir, "final_data.csv")
    header = pd.read_csv(csv, nrows=0).columns
    use = [c for c in columns if c in header] if columns else None
    return pd.read_csv(csv, usecols=use)


def build_reference_marker_classes(
    panel_genes: List[str],
    min_margin: float = 0.2,
) -> pd.DataFrame:
    """Return panel genes with their reference class, built as the pipeline does."""
    from cellscope.pipeline import (
        _configure_rnalocate_augmentation,
        _load_default_gold_tables,
        build_gold_from_rnalocate_l1,
        merge_gold,
        project_l1_to_seven,
    )

    gold_apex, rna_loc = _load_default_gold_tables()
    l1 = build_gold_from_rnalocate_l1(
        rna_loc,
        species=("Homo sapiens", "Mus musculus"),
        agg_mode="prob_or",
        score_floor=0.6,
        use_pubmed_weight=True,
    )
    gold_rna = _configure_rnalocate_augmentation(
        project_l1_to_seven(l1, mode="strict", mito_non_omm=None),
        augment_er=False,
    )
    gold = merge_gold(gold_apex, gold_rna, fuse="max")
    gold.index = gold.index.astype(str).str.upper()
    gold = gold.loc[~gold.index.duplicated()]
    panel = sorted({str(g).upper() for g in panel_genes})
    gold = gold.loc[gold.index.intersection(panel)]

    values = gold.to_numpy(dtype=float)
    order = np.argsort(-values, axis=1)
    top = values[np.arange(len(gold)), order[:, 0]]
    second = values[np.arange(len(gold)), order[:, 1]]
    labels = np.asarray(
        [_GOLD_TO_LABEL.get(c, c) for c in gold.columns], dtype=object
    )[order[:, 0]]
    specific = (top > 0) & ((top - second) >= float(min_margin))
    return pd.DataFrame(
        {
            "gene": gold.index,
            "reference_class": np.where(specific, labels, "unspecific"),
            "top_score": top,
            "margin": top - second,
        }
    )


def cmd_splits(args: argparse.Namespace) -> None:
    genes = pd.read_csv(args.transcripts, usecols=["feature_name"])[
        "feature_name"
    ].astype(str).unique().tolist()
    classes = build_reference_marker_classes(genes, min_margin=args.min_margin)
    os.makedirs(args.out_dir, exist_ok=True)
    classes.to_csv(os.path.join(args.out_dir, "reference_classes.csv"), index=False)
    print("reference genes in panel:", len(classes))
    print(classes["reference_class"].value_counts().to_string())
    for split in range(args.n_splits):
        rng = np.random.default_rng(args.seed + split)
        held = []
        # Stratify by class so each half keeps every structure represented.
        for _, group in classes.groupby("reference_class", sort=True):
            idx = rng.permutation(len(group))
            held.append(group.iloc[idx[: len(group) // 2]])
        heldout = pd.concat(held).sort_values("gene")
        prefix = os.path.join(args.out_dir, f"split{split}")
        heldout["gene"].to_csv(prefix + ".exclude.txt", index=False, header=False)
        heldout[["gene", "reference_class"]].to_csv(
            prefix + ".heldout.csv", index=False
        )
    print(f"wrote {args.n_splits} splits to {args.out_dir}")


def _mh_odds_ratio(
    cell_codes: np.ndarray,
    n_cells: int,
    is_marker: np.ndarray,
    in_region: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Cell-stratified Mantel-Haenszel odds ratio with a cell bootstrap CI."""

    def _count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(cell_codes[mask], minlength=n_cells).astype(np.float64)

    a = _count(is_marker & in_region)
    b = _count(is_marker & ~in_region)
    c = _count(~is_marker & in_region)
    d = _count(~is_marker & ~in_region)
    n = a + b + c + d
    valid = n > 0
    num = np.where(valid, a * d / np.maximum(n, 1), 0.0)
    den = np.where(valid, b * c / np.maximum(n, 1), 0.0)
    result = {
        "marker_transcripts": float(a.sum() + b.sum()),
        "marker_transcripts_in_label": float(a.sum()),
        "background_transcripts_in_label": float(c.sum()),
        "informative_cells": float(((num + den) > 0).sum()),
        "marker_fraction_in_label": float(a.sum() / max(a.sum() + b.sum(), 1)),
        "background_fraction_in_label": float(c.sum() / max(c.sum() + d.sum(), 1)),
    }
    if den.sum() <= 0 or num.sum() <= 0:
        result.update({"mh_or": float("nan"), "ci_low": float("nan"),
                       "ci_high": float("nan"), "boot_p_le_1": float("nan")})
        return result
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_cells, np.full(n_cells, 1.0 / n_cells), size=n_boot)
    boot_num = weights @ num
    boot_den = weights @ den
    boot = np.where(boot_den > 0, boot_num / np.maximum(boot_den, 1e-300), np.nan)
    boot = boot[np.isfinite(boot)]
    result.update(
        {
            "mh_or": float(num.sum() / den.sum()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "boot_p_le_1": float((np.sum(boot <= 1.0) + 1) / (boot.size + 1)),
        }
    )
    return result


def heldout_enrichment(
    final: pd.DataFrame,
    heldout: pd.DataFrame,
    label_col: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    gene = final["feature_name"].astype(str).str.upper().to_numpy()
    held_class = dict(zip(heldout["gene"].str.upper(), heldout["reference_class"]))
    unseen = np.fromiter((g in held_class for g in gene), bool, len(gene))
    labels = final[label_col].astype(str).to_numpy()
    cell_codes, cells = pd.factorize(final["cell_id"], sort=False)
    gene_class = np.asarray([held_class.get(g, "") for g in gene], dtype=object)

    out: Dict[str, Dict[str, float]] = {}
    for fine in FINE_LABELS:
        compartment = NUCLEAR_LABELS if fine in NUCLEAR_LABELS else CYTOPLASMIC_LABELS
        # Only never-seen genes, inside the label's own broad compartment.
        scope = unseen & np.isin(labels, compartment)
        out[fine] = _mh_odds_ratio(
            cell_codes[scope],
            len(cells),
            (gene_class == fine)[scope],
            (labels == fine)[scope],
            n_boot,
            seed,
        )
        out[fine]["heldout_marker_genes"] = float(
            (heldout["reference_class"] == fine).sum()
        )

    # Positive control: do held-out nuclear-class genes sit in nuclear-labelled
    # regions more than held-out cytoplasmic-class genes? This checks that the
    # reference carries spatial signal in this dataset at all.
    classed = np.isin(gene_class, ALL_LABELS)
    out["broad_control_nuclear"] = _mh_odds_ratio(
        cell_codes[classed],
        len(cells),
        np.isin(gene_class, NUCLEAR_LABELS)[classed],
        np.isin(labels, NUCLEAR_LABELS)[classed],
        n_boot,
        seed,
    )
    return out


def insample_enrichment(
    final: pd.DataFrame,
    reference: pd.DataFrame,
    heldout: pd.DataFrame,
    label_col: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Same statistic on genes the annotation DID see (circular upper bound)."""
    seen = reference.loc[
        ~reference["gene"].str.upper().isin(set(heldout["gene"].str.upper()))
    ]
    result = heldout_enrichment(final, seen, label_col, n_boot=n_boot, seed=seed)
    result.pop("broad_control_nuclear", None)
    return result


def evaluate_run(
    run_dir: str,
    heldout_path: Optional[str] = None,
    label_col: str = "subcellular_annotation",
    n_boot: int = 1000,
) -> Dict[str, object]:
    columns = [
        "cell_id", "feature_name", "overlaps_nucleus", "in_nucleus",
        "metaspot", "meta-domain", label_col, "subcellular_annotation_domain",
    ]
    columns = list(dict.fromkeys(columns))
    final = _final_table(run_dir, columns)
    labels = final[label_col].astype(str)
    result: Dict[str, object] = {
        "run_dir": os.path.abspath(run_dir),
        "label_col": label_col,
        "n_transcripts": int(len(final)),
        "n_cells": int(final["cell_id"].nunique()),
        "n_metaspots": int(final["metaspot"].nunique()),
        "n_metadomains": int(final["meta-domain"].nunique()),
    }

    if "overlaps_nucleus" in final.columns:
        truth = pd.to_numeric(final["overlaps_nucleus"], errors="coerce").fillna(0).ne(0)
        broad = {"final_label": labels.isin(NUCLEAR_LABELS)}
        if "in_nucleus" in final.columns:
            broad["polygon_only"] = final["in_nucleus"].astype(bool)
        if "subcellular_annotation_domain" in final.columns:
            broad["metadomain_label"] = (
                final["subcellular_annotation_domain"].astype(str).isin(NUCLEAR_LABELS)
            )
        result["broad_accuracy"] = {
            name: float((pred.to_numpy() == truth.to_numpy()).mean())
            for name, pred in broad.items()
        }

    fractions = labels.value_counts(normalize=True)
    per_cell = final.groupby("cell_id", sort=False)[label_col].agg(
        lambda s: set(s.astype(str))
    )
    result["label_transcript_fraction"] = {
        k: float(fractions.get(k, 0.0)) for k in ALL_LABELS
    }
    result["label_cell_coverage"] = {
        k: float(per_cell.map(lambda s, k=k: k in s).mean()) for k in ALL_LABELS
    }
    result["other_label_fraction"] = float(1.0 - labels.isin(ALL_LABELS).mean())

    if heldout_path:
        heldout = pd.read_csv(heldout_path)
        result["heldout_table"] = os.path.abspath(heldout_path)
        result["heldout_enrichment"] = heldout_enrichment(
            final, heldout, label_col, n_boot=n_boot
        )
        reference_path = os.path.join(
            os.path.dirname(os.path.abspath(heldout_path)), "reference_classes.csv"
        )
        if os.path.isfile(reference_path):
            result["insample_enrichment"] = insample_enrichment(
                final, pd.read_csv(reference_path), heldout, label_col, n_boot=n_boot
            )
    return result


def cmd_evaluate(args: argparse.Namespace) -> None:
    result = evaluate_run(args.run_dir, args.heldout, args.label_col, args.n_boot)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


def cmd_compare(args: argparse.Namespace) -> None:
    from sklearn.metrics import adjusted_rand_score

    columns = ["metaspot", "meta-domain", args.label_col]
    a = _final_table(args.a, columns)
    b = _final_table(args.b, columns)
    if len(a) != len(b):
        raise SystemExit("runs have different transcript counts")
    result = {
        "label_agreement": float(
            (a[args.label_col].astype(str).to_numpy()
             == b[args.label_col].astype(str).to_numpy()).mean()
        ),
        "label_ari": float(adjusted_rand_score(
            a[args.label_col].astype(str), b[args.label_col].astype(str))),
        "metaspot_ari": float(adjusted_rand_score(a["metaspot"], b["metaspot"])),
        "metadomain_ari": float(adjusted_rand_score(a["meta-domain"], b["meta-domain"])),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("splits")
    p.add_argument("--transcripts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2026)
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--heldout")
    p.add_argument("--label-col", default="subcellular_annotation")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-col", default="subcellular_annotation")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
