# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/en/1.0.0/) and this project adheres to Semantic Versioning.

## [1.0.0] - 2026-11-18
First release under the name CellScoper; the packaged defaults are the configuration of the v1.0.0 results.
### Changed (2026-11-17/18)
- Default label rule `annotation.evidence_label_rule: lens_count` (`cellscoper/lens_count.py`): a gate on held-out
  reference markers in the LENS decoder, the amount per cell side from LENS gene weights (one Otsu threshold per
  structure), and placement within the cell by evidence and structure-axis ranks. `axis` remains available.
- LENS defaults `module5.lens_background: own` (predict hidden genes against the cell side's own RNA) and
  `module5.lens_absolute_position: false`.
- `module1.remove_foreign_transcripts` stays on: on Atera BRCA 1k, switching it off left label accuracy unchanged
  (held-out genes) but produced a MetaDomain made of fibroblast extracellular-matrix RNA from neighbouring cells.
### Added (2026-11-18)
- Diagnostic figures in `figures/` (01-09: foreign-transcript removal, MetaSpots, gene embedding, LENS training, LENS
  against gene-shuffled cells, MetaDomains with their most enriched observed genes, annotation validation, label share
  against RNA depth, example cells); a text-overlap check runs before each figure is saved.
- LENS training curve `intermediate_results/module5_lens_training.csv` (train NLL, held-out gain and time per epoch)
  and the kept epoch, stop reason and data split in `adata1.uns["lens"]`.

## Earlier changes (2026-11-03)
### Changed (2026-11-03 cleanup)
- The package is renamed `cellscope` -> `cellscoper` (imports, `cellscoper` command, `CELLSCOPER_*` environment
  variables, logger name). The `cellscope` command alias is removed.
- `pipeline.py` is split into modules: `runtime`, `keys`, `checkpoints`, `outputs`, `geometry` (Module 1),
  `composition` (Module 2), `metaspots` (Modules 3-4), `metadomains` (Modules 5-6), `reference` and the orchestration in
  `pipeline`. Outputs on Atera BRCA 1k are bit-identical to the version before the cleanup.
- Defaults (2026-11-02/03): structure-axis annotation on GE + LENS; the cell x structure table on.
- MetaDomains (`meta-domain`) are now found from LENS without reference genes (HDBSCAN against gene-shuffled cells,
  `module5.metadomain_q`; background `-1`), replacing the 25 MiniBatchKMeans clusters per side; `adata2` and
  `subcellular_annotation_domain` follow them. The separate `metadomain_discovered` column, the GE + LENS concatenation
  `X_ge_lens` and the parameters `fixed_k`, `min_domain_size`, `train_sample_per_block`, `assign_chunk_size`, `mbk_*`,
  `lens_weight`, `metadomain_discovery` and `metadomain_discovery_q` are removed.
- Module 3 parameters renamed: `hdbscan_min_cluster_size` -> `min_segment_points`, `n_feat_hdb` ->
  `n_segmentation_features`.
- The masked-transcript model is named LENS throughout: module `cellscoper/lens.py` (`lens_embedding`), parameters
  `module5.lens_max_epochs`, `lens_spots_per_epoch`, `lens_null_cells` (were `mtm_*`), outputs `obsm["X_lens"]` and
  `uns["lens"]` (were `X_metaspot_mtm`, `uns["metaspot_mtm"]`).
- MetaDomain discovery combines 5 restarts (different HDBSCAN samples and cell halves); a direction must be kept in at
  least 3 of them (`uns["metadomain_discovery"]["support"]`). Atera 5k: 4 MetaDomains for every LENS and discovery seed
  tried (before: 0-4 depending on the discovery seed).
- `evidence_annotation_report.json` records each structure's read-out weights (`axis_weights`: GE per scale, GE,
  LENS and LENS's share); labels are unchanged. Atera 1k: LENS share 32-56%.
### Removed (2026-11-03 cleanup)
- CA-DMGI (Modules 7-8, `multiplex_dgi`, `sampled_multiplex_dgi`, `scalable_graph`) and all `module7` / `module8`
  parameters; kNN backends and DGI training modes in the execution plan; MetaDomain pooling of representations
  (`module6.pool_reps`).
- The count-based annotation rule (`evidence_label_rule: legacy`, `evidence_score`, `evidence_unit`, `evidence_fdr`,
  `evidence_null_sets`, `evidence_label_unit`).
- Alternative MetaDomain clustering (consensus Leiden, hierarchical prototypes, automatic K, per-cell cap, cosine,
  column standardisation, other `cluster_rep` choices) and their parameters.
- LENS options not adopted (`mtm_fixed_tokens`, `mtm_smooth_k`, `mtm_shrink_background`, `mtm_include_absolute_position`,
  `mtm_bind_decoder_to_ge`; `mtm_exclude_visible` is always on).
- Module 3 HDBSCAN / KMeans segmentation (`enforce_spatial_connectivity`, `partition_by_nucleus`, `segmentation_features`
  and the `hdbscan_*` / `kmeans_*` options); MetaSpots are always spatially constrained Ward per cell side.
- Module 2 non-streamed and cell-summary paths (`stream_local_embeddings`, `source_from_point_df`,
  `CELLSCOPE_CELL_SUMMARY_ONLY`, `CELLSCOPE_ULTRALOW`).
- The marker-composition cell features (`compute_cell_features`), `gene_embedding_scale: colocalisation`,
  `benchmarks/`, `benchmark_utils`, `plotting_style`, `config/params.fast.yaml`, `src/manual_marker.csv`, migration of
  old checkpoint column names (`cell_sub`, `cluster_in_cell`) and other compatibility shims.
### Changed
- The default reference now also contains `src/extra_reference_markers.tsv`
  (CAP-seq ER, LoRNA mitochondrion fraction for OMM, nuclear-speckle APEX-seq).
  This adds an eighth label, `Nuclear Speckle`. Held-out enrichment of ER, OMM
  and Nuclear Speckle improved on both test datasets.
- A structure without any marker gene in the panel is no longer placed.
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
