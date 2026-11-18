# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project follows Semantic Versioning.

## [1.0.0] - 2026-11-18

First release under the name CellScoper. The repository was previously named CellScope; this release replaces that
code. The packaged defaults are the configuration of the v1.0.0 results.

### Changed compared with the earlier CellScope code
- The package is `cellscoper` and the command is `cellscoper` (was `cellscope`).
- Representation: the CA-DMGI graph model is replaced by a spatial gene embedding (GE) and LENS, a masked-transcript
  model that learns where each MetaSpot's RNA departs from its own cell side.
- MetaDomains are found from LENS without reference genes: modes that exceed gene-shuffled cells and recur in both
  halves of the cells and across restarts; their number comes from the data. They replace fixed-number KMeans clusters.
- Annotation (`evidence_label_rule: lens_count`): a structure is labelled only if its held-out reference markers pass a
  gate against random gene sets; LENS gene weights set the amount per cell side and evidence and structure-axis ranks
  set the placement. Every labelled structure is validated on held-out reference markers. When a dataset cannot
  resolve the nucleus from the cytoplasm, only `Nucleus`/`Cytosol` are reported.
- Transcripts foreign to the cell's expression group (mostly segmentation spillover) are removed by default
  (`module1.remove_foreign_transcripts`).

### Added
- Cell × structure table (`cell_structure_features.parquet`): per cell side, label amounts and a pair co-localisation
  ratio.
- Diagnostic figures (`figures/01–09`) and a generated `OUTPUT_GUIDE.md` in every output folder; the LENS training
  curve (`intermediate_results/module5_lens_training.csv`).
- Scale-aware execution for whole sections: streaming profile, partitioned final table and sampled LENS training.

### Removed
- CA-DMGI (Modules 7–8), consensus Leiden and hierarchical MetaDomain clustering, the count-based annotation rule,
  the benchmark helpers and the `cellscope` command alias.
