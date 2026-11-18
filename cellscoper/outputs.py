"""Final outputs: the OUTPUT_GUIDE.md index, the cell-aligned partitioned final table, and compact h5ad writing."""

from contextlib import contextmanager
from typing import Any, Dict, Mapping
import hashlib
import math
import os
import tempfile
import anndata as ad
import numpy as np
import pandas as pd

from .checkpoints import _intermediate_dirname, _plain_strings
from .keys import LENS_REP


def _atomic_partitioned_parquet_write(
    df: pd.DataFrame,
    directory: str,
    *,
    target_rows_per_part: int = 1_000_000,
) -> Dict[str, Any]:
    """Write a cell-stable Parquet dataset with an atomic directory swap."""
    import json
    import shutil

    parent = os.path.dirname(os.path.abspath(directory))
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(
        prefix=f".{os.path.basename(directory)}.",
        suffix=".tmp",
        dir=parent,
    )
    target_rows = max(1, int(target_rows_per_part))
    n_parts = max(1, int(math.ceil(len(df) / target_rows)))
    if "cell_id" in df.columns:
        # Hash the normalized cell identifier itself, rather than its first
        # occurrence in this frame. This keeps a cell in the same partition
        # when input rows are reordered and makes manifest identities resumable.
        cell_ids = df["cell_id"].astype(str).to_numpy()
        cell_hashes = np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(cell_id.encode("utf-8"), digest_size=8).digest(),
                    byteorder="little",
                    signed=False,
                )
                for cell_id in cell_ids
            ),
            dtype=np.uint64,
            count=len(cell_ids),
        )
        partition_ids = np.mod(cell_hashes, n_parts).astype(np.int32)
    else:
        partition_ids = np.minimum(
            np.arange(len(df), dtype=np.int64) // target_rows,
            n_parts - 1,
        ).astype(np.int32)
    order = np.argsort(partition_ids, kind="stable")
    sorted_parts = partition_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_parts)) + 1]
    ends = np.r_[starts[1:], len(order)]
    parts = []
    try:
        for start, end in zip(starts, ends):
            part_id = int(sorted_parts[start]) if end > start else 0
            name = f"part-{part_id:05d}.parquet"
            path = os.path.join(temporary, name)
            frame = df.iloc[order[start:end]]
            _plain_strings(frame).to_parquet(path, index=False)
            parts.append(
                {
                    "file": name,
                    "rows": int(len(frame)),
                    "cells": (
                        int(frame["cell_id"].nunique())
                        if "cell_id" in frame.columns
                        else None
                    ),
                }
            )
        manifest = {
            "schema_version": 1,
            "format": "partitioned_parquet",
            "partition_strategy": (
                "stable_cell_hash" if "cell_id" in df.columns else "row_range"
            ),
            "rows": int(len(df)),
            "columns": [str(column) for column in df.columns],
            "n_parts": int(len(parts)),
            "target_rows_per_part": target_rows,
            "parts": parts,
        }
        with open(os.path.join(temporary, "manifest.json"), "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        open(os.path.join(temporary, "_SUCCESS"), "w").close()
        previous = f"{directory}.previous"
        if os.path.exists(previous):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            else:
                os.unlink(previous)
        if os.path.exists(directory):
            os.replace(directory, previous)
        os.replace(temporary, directory)
        if os.path.exists(previous):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            else:
                os.unlink(previous)
        return manifest
    finally:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)

def _write_output_guide(
    out_dir: str,
    *,
    counts: Mapping[str, Any],
    annotation_enabled: bool,
    primary_data_product: str = "final_data.parquet",
) -> str:
    """Write a concise human-readable index of production and audit outputs."""
    intermediate_name = _intermediate_dirname()
    guide_path = os.path.join(out_dir, "OUTPUT_GUIDE.md")
    count_text = (
        f"{int(counts.get('n_transcripts') or 0):,} transcripts, "
        f"{int(counts.get('n_cells') or 0):,} cells, "
        f"{int(counts.get('n_metaspots') or 0):,} MetaSpots, and "
        f"{int(counts.get('n_metadomains') or 0):,} MetaDomains"
    )
    annotation_section = ""
    if annotation_enabled:
        annotation_section = f"""
## Subcellular annotation

- `subcellular_annotation`: the final label. Each MetaSpot has one continuous
  score per structure, read from the spatial gene embedding and LENS along the
  direction of the structure's reference markers and averaged over
  neighbouring MetaSpots of its cell side; a MetaSpot is labelled only when its
  score clears a threshold set on gene-shuffled cells at the target share of
  false labels (`evidence_axis_q`). A structure that never beats the shuffle is
  not labelled anywhere, so cells need not show every structure.
- `subcellular_score__<structure>`, `subcellular_q__<structure>`: the
  continuous structure score of each MetaSpot and the estimated share of false
  labels at that score. `subcellular_annotation_placed` shows every
  structure's best MetaSpot per cell, for display only.
- `subcellular_annotation_support`: how far each final label can be trusted.
  `segmentation`: Nucleus/Cytosol, decided by the nuclear boundary.
  `validated`: the structure passed held-out marker validation in this dataset;
  such regions carry more of its marker RNA than the rest of the cell, typically
  by a few tens of percent, so read them as "enriched for", not "is".
  `unvalidated`: labelled, without held-out support in this dataset; do not
  base conclusions on these.
- `subcellular_annotation_validated`: the final label restricted to
  structures that passed held-out validation; everything else is `Nucleus` or
  `Cytosol`. `subcellular_annotation_exploratory`: the same with a relaxed
  validation criterion.
- `evidence_annotation_report.json`: per-structure threshold, labelled
  MetaSpots, held-out marker enrichment with confidence interval and whether
  it is supported; `unsupported_structures` lists labelled structures without
  held-out support.
- `meta-domain`: MetaDomain found from LENS without reference genes
  (HDBSCAN against gene-shuffled cells), -1 = background; per-MetaDomain side,
  nucleus share, top genes and nucleus-boundary flag in
  `adata1.uns["metadomain_discovery"]`. A flagged MetaDomain follows the
  nucleus-cytoplasm contrast across the nucleus outline.
- `cell_structure_features.parquet` (with `io.cell_structure_features`):
  one row per cell side and structure / MetaDomain: label shares (fall with
  depth) and the pair co-localisation ratio `concentration` (no depth bias,
  noisy per cell: compare groups by sum(`pair_obs`) / sum(`pair_exp`)).

Comparing cells: use `cell_structure_features.parquet`. A cell's labelled
share of a structure is its amount, judged by the same threshold in every cell;
it rises with transcript depth, so compare cell types at matched depth. The
pair co-localisation ratio measures how concentrated the structure's RNA is
inside the cell, has no depth bias, and does not rise when a structure fills
the cell (Atera 5k: plasma cells, ER). The `subcellular_score__*` columns are
relative to the cell's own RNA mixture.
"""
    primary_name = os.path.basename(primary_data_product)
    optional_name = (
        "final_data.csv"
        if primary_name == "final_data.parquet"
        else "final_data.parquet"
    )
    text = f"""# CellScoper output guide

This successful run contains {count_text}.

## Primary data products

- `{primary_name}`: transcript-level production result written successfully by
  this run. `{optional_name}` is optional and must not be treated as current
  unless it is also recorded in `run_manifest.json`.
- `adata1.h5ad`: MetaSpot-level production object with the same annotation
  columns in `obs`; the validation report is also stored in
  `uns[\"evidence_annotation\"]`.
- `adata2.h5ad`: internal MetaDomain evidence and clustering object; it is not
  the final annotation grain.
- `effective_params.yaml`, `params_used.yaml`, and `run_manifest.json`:
  configuration and provenance required to reproduce the run.
{annotation_section}
## Diagnostic figures (`figures/`, with `io.write_intermediate_figures`)

What each step did, in pipeline order. They summarise this run for checking;
the numbers behind them are in the files above.

- `01_foreign_transcripts.png`: transcripts per cell before and after removing
  genes foreign to the cell's expression group, removed share per cell and per
  internal group (k-means on expression, not a cell-type annotation).
- `02_metaspots.png`: MetaSpots per cell, transcripts per MetaSpot, and
  MetaSpots against cell depth.
- `03_gene_embedding.png`: mean similarity of the spatial gene embedding
  between reference marker sets (diagonal = within one set) and within each set
  against random gene sets of equal size; a high diagonal means the embedding
  captured the markers' co-localisation.
- `04_lens_training.png` (table: `{intermediate_name}/module5_lens_training.csv`):
  on held-out cells, how much more likely LENS makes the true hidden gene than
  a guess from the cell's average RNA (%), per epoch, and the training loss;
  the kept epoch is marked.
- `05_lens_signal.png`: share of MetaSpots with a strong LENS signal in real
  cells against the same cells with genes shuffled inside each cell side; along
  each structure's direction (what the labels use), how much more real scores
  vary than shuffled ones and how little they share with them (MetaSpot size
  and position); and real minus shuffled signal in one example cell.
- `06_metadomains.png`: size, nucleus share and top genes of each MetaDomain;
  share above threshold in real against gene-shuffled cells.
- `07_annotation_validation.png`: held-out marker enrichment with 95% CI and
  pass/fail per structure, share of MetaSpots per final label, held-out marker
  AUROC against random gene sets.
- `08_structure_share_vs_depth.png`: labelled share against RNA depth per cell
  side; the rise with depth is why cells are compared at matched depth.
- `09_example_cells.png`: six cells chosen by RNA depth only (10–97%
  quantiles): transcripts with cell and nucleus outlines, MetaSpots, final
  labels and MetaDomains.

## Checkpoints and compact QC

Other files under `{intermediate_name}/` are restart checkpoints or compact
diagnostic tables. They should not be presented as final biological results
unless their observation unit and decision stage are stated explicitly.
"""
    fd, temporary = tempfile.mkstemp(
        prefix=".OUTPUT_GUIDE.",
        suffix=".tmp.md",
        dir=out_dir,
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text.strip() + "\n")
        os.replace(temporary, guide_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return guide_path

@contextmanager
def _compact_final_adata1_for_write(
    adata1: ad.AnnData,
    *,
    enabled: bool,
):
    """Temporarily retain only the LENS embedding (X_lens) of the MetaSpot representations.

    Final output compaction must not mutate the in-memory object or any resume
    checkpoint: the pipeline may still need all relations while writing later
    artifacts.  Holding references while the mappings are cleared avoids a
    second multi-GiB AnnData copy.
    """
    if not enabled:
        yield adata1
        return

    original_obsm = list(adata1.obsm.items())
    original_obsp = list(adata1.obsp.items())
    kept_obsm = [
        (key, value)
        for key, value in original_obsm
        if key == LENS_REP
    ]
    marker_key = "cellscoper_final_output_compaction"
    sentinel = object()
    original_marker = adata1.uns.get(marker_key, sentinel)
    adata1.obsm.clear()
    adata1.obsm.update(kept_obsm)
    adata1.obsp.clear()
    adata1.uns[marker_key] = {
        "enabled": True,
        "kept_obsm": [key for key, _ in kept_obsm],
        "omitted_obsm": [
            key for key, _ in original_obsm if key != LENS_REP
        ],
        "omitted_obsp": [key for key, _ in original_obsp],
        "reason": "io.drop_embeddings",
    }
    try:
        yield adata1
    finally:
        adata1.obsm.clear()
        adata1.obsm.update(original_obsm)
        adata1.obsp.clear()
        adata1.obsp.update(original_obsp)
        if original_marker is sentinel:
            adata1.uns.pop(marker_key, None)
        else:
            adata1.uns[marker_key] = original_marker
