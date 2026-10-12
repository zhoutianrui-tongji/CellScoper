# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.0.0/) and this project adheres to Semantic Versioning.

## [Unreleased]
### Changed
- `module1.remove_foreign_transcripts` is on by default.
- New column `subcellular_annotation_support` grades every final label as
  `segmentation`, `validated` or `unvalidated`.
- Validated structures take every unit that scores clearly above the cell's
  average, not only one unit per cell.
- `subcellular_annotation` is now the seven-structure label (previously
  `subcellular_annotation_complete`). The labels restricted to structures that
  pass held-out validation moved to `subcellular_annotation_validated`.
### Removed
- The RAP-X annotation and its MetaSpot refinement, rescue and OMM refinement
  branches, with their 150+ `annotation.*` parameters, audit tables and
  figures. `annotation.method: evidence` is now the only method and the
  default; `method: rapx` raises an error. The last version containing RAP-X
  is the git tag `legacy-rapx`. Columns that only RAP-X produced (for example
  `subcellular_confidence`, `subcellular_annotation_provenance`,
  `subcellular_metaspot_*`, `omm_*`) are no longer written.
### Fixed
- Consensus Leiden resolution selection no longer flips between the finest
  and coarsest candidate on identical inputs: near-best candidates are treated
  as ties and `consensus_min_stability` is reported (`stability_floor_met`,
  log warning) instead of gating the choice.
- Module 1 and checkpoint fingerprinting no longer swallow configuration
  errors and fall back to code defaults that differ from `params.yaml`.
- The CLI run summary opens result `.h5ad` files in backed mode instead of
  loading them into memory.
- RNALocate "nuclear lamina"/"nuclear membrane" and mitochondrial
  ribosome/matrix terms map to the correct localization classes.
- Batched RAP-X, hybrid MetaSpot, provenance, and transcript annotation column
  writeback to avoid pandas DataFrame fragmentation and repeated
  `PerformanceWarning` messages on annotation-heavy runs.
- Preserved nucleus-contour vertex order and explicitly closed polygons so
  boundary-derived nuclear membership is not corrupted by row sorting.
- Prevented changed input files from reusing stale raw-input checkpoints.
- Removed `overlaps_nucleus`-based cell selection when polygon geometry is
  configured, avoiding benchmark population leakage.
- Made malformed explicit YAML configurations fail visibly.

### Added
- `subcellular_annotation_complete` (evidence annotation): all seven structures
  placed in every cell at their best-scoring unit, independent of validation.
- `module1.remove_foreign_transcripts` drops transcripts whose
  gene is foreign to the cell's type, i.e. likely spillover from a neighbouring
  cell, before MetaSpots are built.
- `annotation.method: evidence`: an alternative annotation with ten
  parameters. Units are scored against a within-cell mixing expectation,
  tested against a within-cell permutation null, and each structure is
  validated on held-out marker genes; only structures supported in the current
  dataset enter `subcellular_annotation`. A resolution gate reports
  Nucleus/Cytosol only when gene positions carry no reproducible signal.
  It is now the only annotation method.
- `module1.boundary_distance: edge` for exact point-to-polygon distances.
- `annotation.reference_exclude_genes_file` and `benchmarks/evaluate_run.py`
  for held-out marker validation of annotation labels.
- Native CSV/Parquet input loading and bounded previews for all three tables.
- Packaged default configuration and annotation references for wheel installs.
- Added Cell-Aware Deep Multiplex Graph Infomax (CA-DMGI) with separate
  cross-cell expression and
  within-cell spatial encoders, within-cell feature corruption, a bounded
  node-wise consensus gate, branch embeddings, and full training provenance.
- Added sparse stability-selected consensus Leiden for MetaDomain discovery,
  including multi-resolution/multi-seed auditing, cross-cell recurrence
  constraints, and edge-level co-association without a dense consensus matrix.

### Changed
- `module3.partition_by_nucleus` now defaults to `true` so a MetaSpot cannot
  straddle the nuclear boundary (2k-cell Xenium audit: nuclear/cytoplasmic
  accuracy 0.896 -> 0.951, equal to the polygon-only ceiling).
- Reduced CA-DMGI full-batch cost by fusing positive/corrupted sparse
  propagation, auto-tuning CPU threads, using compact CSR indices, exposing an
  explicit CPU/CUDA selector with fail-fast CUDA validation, and skipping the
  unused standard-DGI graph conversion on the multiplex path.
- Made CA-DMGI plus consensus Leiden the packaged unsupervised
  clustering path; the previous single-graph DGI and MiniBatchKMeans paths
  remain available for reproducible ablations.
- Increased fragmentation-aware OMM sensitivity: pooled cell evidence still
  requires at least 6 marker transcripts from 3 genes plus enrichment and a
  binomial gate, while a focal one-marker anchor can be localized only with
  OMM-over-ER/Cytosol separation and majority DGI support. The bounded per-cell
  cap is now 5 evidence-qualified loci.
- Replaced the default two-layer plain DGI-GCN encoder with a configurable
  two-block initial-residual encoder and changed the packaged MetaSpot DGI
  output dimension from 64 to 32. The historical plain encoder remains
  available through `module8.dgi_encoder: plain`.
- Added DGI architecture, depth, dropout, residual, normalization, optimizer,
  and loss-history provenance; architecture changes now invalidate M8 caches.
- Renamed the user-facing project and CLI to CellScoper; `cellscope` remains a compatibility alias.
- Replaced the output columns `cluster_in_cell` and `cell_sub` with `metaspot_in_cell` and `metaspot`.
- Made spatially constrained Ward clustering with an adaptive unique-XY-site target the canonical MetaSpot segmenter.
- Restored the 200,000-node full-batch MetaSpot-DGI safety guard and documented
  `metadomain` as the large-dataset path.
- Limited warning suppression to targeted third-party compatibility notices.

## [1.0.0] - 2025-12-02
### Added
- Initial public release of CellScoper.
- Modular pipeline (M1–M8), YAML configuration, resumable checkpoints.
- Clear module completion/cache messages; first-run suffix suppression.
- Intermediate save notifications and configurable intermediate directory.
- RAP-X annotation with RNALocate/APEX mapping and diagnostics.
- Comprehensive configuration reference under `docs/configuration.md`.

### Fixed
- CLI bin script shebang correctness.

### Changed
- README rewritten in English with concise quickstart.
