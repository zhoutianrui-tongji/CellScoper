# CellScoper

Subcellular structure discovery and annotation for imaging-based spatial transcriptomics.

CellScoper finds subcellular structures in segmented cells from transcript positions alone, labels known structures
with built-in validation, and reports per-cell structure amounts. It takes transcript coordinates with cell and nucleus
segmentation and has been run on 10x Xenium, Atera, CosMx and a HEK293T Xenium panel. The repository provides a CLI and
a modular pipeline with resume and lightweight checkpoints.

With the packaged `config/params.yaml` the pipeline runs these steps:

1. **Preprocessing** keeps cells present in all three input tables and removes transcripts whose gene the cell's type
   barely expresses, which are mostly segmentation spillover from neighbouring cells
   (`module1.remove_foreign_transcripts`).
2. **M1** computes per-transcript geometry against the cell and nucleus polygons, including nuclear membership.
3. **M2** summarises the gene composition around each transcript (distance-weighted genes of its 12 nearest
   neighbours, TF-IDF, sparse random projection).
4. **M3** splits each side of every cell (nucleus, cytoplasm) into spatially connected MetaSpots by constrained Ward
   clustering on the M1/M2 features plus coordinates (3D when `z_location` is given), about one MetaSpot per 15
   transcript sites. **M4** collects them in `adata1`.
5. **M5** builds the MetaSpot representation and the MetaDomains. The representation has two parts: a spatial gene
   embedding (GE) learned from within-cell gene co-localisation, and LENS, a masked-transcript model that learns from
   hidden genes where each MetaSpot's RNA departs from its own cell side (`cellscoper/lens.py`; background = the cell
   side's own RNA, `module5.lens_background: own`; no absolute radial position, `module5.lens_absolute_position: false`;
   a CUDA GPU is recommended). **MetaDomains** are found from LENS without reference genes (`cellscoper/metadomain_discovery.py`):
   directions of the LENS embedding that real MetaSpots take far more often than gene-shuffled cells, found by
   density (HDBSCAN, nucleus and cytoplasm together) and kept when both halves of the cells show them. Their number
   comes from the data; other MetaSpots are background (`meta-domain = -1`). **M6** pools MetaDomains into `adata2`.
6. **Annotation** (`annotation.evidence_label_rule: lens_count`, `cellscoper/lens_count.py`) labels known structures
   in three steps. *Gate*: a structure is labelled only if, in the LENS decoder, held-out reference markers rank above
   expression-matched random gene sets (z >= 3). *Amount*: each gene gets a weight along the structure's marker
   direction; a MetaSpot's evidence is the mean weight of its transcripts, and one Otsu threshold per structure over all
   MetaSpots of its side sets how many transcripts of each cell side count for the structure. *Place*: within each cell
   side, MetaSpots are ranked by the mean of two percentile ranks, this evidence and the structure-axis score
   (`cellscoper/structure_axes.py`: the marker direction in GE and LENS, averaged over neighbouring MetaSpots), and the
   structure takes MetaSpots from the top until its amount is filled. Held-out reference markers then test every
   labelled structure. References: RNALocate v3, APEX-seq and `src/extra_reference_markers.tsv` (CAP-seq ER,
   multi-method OMM, nuclear structures from genome position). The earlier structure-axis rule (labels above a
   threshold set on gene-shuffled cells) remains available as `evidence_label_rule: axis`.
7. **Cell x structure table** (`cellscoper/cell_features.py`): per cell side, structure label and MetaDomain, the labelled share of MetaSpots and transcripts (amount; compare at matched transcript depth) and a pair
   co-localisation ratio (arrangement; no depth bias; compare groups with sum(`pair_obs`) / sum(`pair_exp`)).

No subcellular gold label is used to train the representation or to find the MetaDomains.

## System Requirements
**Operating System**: Linux (tested on Rocky Linux 9.4)

**Python dependencies**: numpy, scipy, pandas, scikit-learn (>= 1.3), anndata, shapely, pyarrow, joblib, matplotlib,
PyYAML, threadpoolctl, click, rich and torch. See `requirements.txt`.

**Hardware**: a CUDA GPU for LENS (a 24 GB card was used for all runs below). 1,000 cells take a few minutes; the full
Atera BRCA section (168,236 cells, 479 million transcripts) took 7 h 13 min with 256 GiB peak memory.

## Install CellScoper
### Recommended (Conda environment)
```bash
git clone https://github.com/zhoutianrui-tongji/CellScoper.git
cd CellScoper
conda create -n cellscoper-env python=3.9
conda activate cellscoper-env
pip install -r requirements.txt
pip install -e .
```

## Documentation
Configuration and usage are documented in this `README.md` and the `docs/` folder. The main configuration is `config/params.yaml` (I/O, module parameters, annotation settings).

## Usage
```bash
cellscoper --help
Usage: cellscoper [OPTIONS]

Options:
  --version                    Show the version and exit.
  --spatial PATH               Transcripts CSV/Parquet  [required]
  --cell-boundaries PATH       Cell boundaries CSV/Parquet  [required]
  --nucleus-boundaries PATH    Nucleus boundaries CSV/Parquet  [required]
  --out-dir PATH               Output directory  [required]
  --gold-data PATH             Gold table CSV/Parquet replacing the default
                               annotation reference
  --no-resume                  Disable resume from checkpoints
  --no-save-intermediate       Do not save intermediate checkpoints
  --no-show-sample             Do not print input sample rows
  --config FILE                YAML merged over config/params.yaml
  --internal-progress / --no-internal-progress
                               Show internal step prints and timing
  --resume-policy [auto|minimal|force]
                               Resume behavior (default: io.resume_policy)
  --resume-from [load_inputs|preprocess|build_anndata|annotate|write_outputs]
                               Stage to rerun from
  --dry-run-diff               Show which stage would rerun; do not execute
  -h, --help                   Show this message and exit.
```
Example:
```bash
cellscoper \
  --spatial sample/spatial_transcripts.csv \
  --cell-boundaries sample/cell_boundaries.csv \
  --nucleus-boundaries sample/nucleus_boundaries.csv \
  --out-dir sample_out
```

`--config` files only need the keys they change; the packaged defaults are the configuration of the v1.0.0 results.
`config/params.atera_brca.yaml` is an earlier option with finer MetaSpots (6 transcript sites per MetaSpot); the
v1.0.0 results use the default 15.

## Inputs

CellScoper expects three primary tabular inputs (CSV or Parquet), with required columns as below.

1) Transcripts table (`--spatial`)
- Required columns: `cell_id`, `x_location`, `y_location`, `feature_name`
- Meaning:
	- `cell_id`: the cell identifier each transcript belongs to
	- `x_location`/`y_location`: transcript coordinates in image/pixel space
	- `feature_name`: gene or feature name of the transcript (e.g. `ACTB`)
	- `z_location`: optional transcript height (same unit as x/y, e.g. Xenium
	  `transcripts.parquet`); when present, the spatial gene embedding finds
	  neighbours in 3D, which separates structures better on Atera BRCA.
	- `overlaps_nucleus`: optional. Nuclear membership is inferred from the
	  nucleus polygons; the column is required only with
	  `module1.use_input_nucleus_flags: true` (values `0/1` or `true/false`).
- Example (CSV):
	```csv
	cell_id,x_location,y_location,feature_name
	C001,102.3,55.8,ACTB
	C001,98.1,60.2,GAPDH
	C002,210.0,120.5,TUBB
	```

2) Cell boundaries (`--cell-boundaries`)
- Required columns: `cell_id`, `vertex_x`, `vertex_y`
- Meaning:
	- Each row is one polygon vertex of a cell boundary; vertices with the same `cell_id` are grouped to form the polygon.
	- Vertex ordering should trace the boundary (clockwise or counter‑clockwise) for each `cell_id`.
- Example (CSV):
	```csv
	cell_id,vertex_x,vertex_y
	C001,90.0,40.0
	C001,130.0,40.0
	C001,130.0,80.0
	C001,90.0,80.0
	C002,200.0,100.0
	C002,240.0,100.0
	C002,240.0,140.0
	C002,200.0,140.0
	```

3) Nucleus boundaries (`--nucleus-boundaries`)
- Schema identical to cell boundaries: `cell_id`, `vertex_x`, `vertex_y`
- Vertices with the same `cell_id` define the nucleus polygon of that cell.
- Example (CSV):
	```csv
	cell_id,vertex_x,vertex_y
	C001,100.0,50.0
	C001,120.0,50.0
	C001,120.0,70.0
	C001,100.0,70.0
	```

Notes:
- Column names are fixed as above; validation reports missing required columns.
- CSV/Parquet are auto‑detected; top‑5 sampling previews print unless `--no-show-sample`.
- Coordinates should be in the same reference frame for transcripts and boundaries.

## Scale-aware execution

`execution.profile: auto` routes from the realized MetaSpot count: up to 2,000,000 MetaSpots keeps full diagnostic
tables; larger runs (`streaming`, e.g. full Atera with ~24M MetaSpots) skip them and, with
`io.write_partitioned_dataset: true`, also write `final_data_dataset/`, a cell-aligned partitioned Parquet dataset with
`manifest.json` and `_SUCCESS`. The resolved plan is recorded in `intermediate_results/execution_plan.json` and in
`adata1.uns`. For LENS on very large runs, `module5.lens_spots_per_epoch` trains each epoch on a fixed number of random
MetaSpots, so training time does not grow with the data. The full Atera run above added only these two keys to the
defaults:

```yaml
module5:
  lens_spots_per_epoch: 1000000
io:
  write_partitioned_dataset: true
```

## Outputs

- `adata1.h5ad`: MetaSpot-level object. `obs`: `subcellular_annotation` (final label: a structure placed by the
  annotation step, else `Nucleus`/`Cytosol`), `subcellular_annotation_support`
  (`segmentation`, `validated` or `unvalidated`), `subcellular_annotation_validated` (only structures that passed
  held-out validation in this dataset), `subcellular_annotation_exploratory` (relaxed validation),
  `subcellular_annotation_placed` (every structure's best MetaSpot per cell, display only), the MetaDomain summary
  `subcellular_annotation_domain`, per-structure `subcellular_score__*` / `subcellular_q__*`, `meta-domain` (-1 = background) and
  `meta-domain_score`. `obsm`: `X_gene_embedding` and `X_lens` (LENS); `uns["metadomain_discovery"]`: side, nucleus
  share, decoder gene profile, top genes and nucleus-boundary flag of each MetaDomain. With
  `io.drop_embeddings: true` only `X_lens` is kept.
- `adata2.h5ad`: MetaDomain-level object from M6 (counts and geometry summaries; background is the row `-1`).
- `final_data.csv|parquet`: transcript-level results with `metaspot_in_cell`, `metaspot`, `meta-domain` and the
  annotation columns.
- `cell_structure_features.parquet`: the cell x structure table (step 7).
- `evidence_annotation_report.json`: per-structure threshold, labelled MetaSpots, held-out marker enrichment with
  confidence interval and whether the structure is supported in this dataset.
- `effective_params.yaml`, `params_used.yaml`, `run_manifest.json`: exact effective configuration and
  run/input/reference/code provenance.
- `OUTPUT_GUIDE.md`: generated index of the outputs, including what each diagnostic figure shows.
- `figures/01–09_*.png`: diagnostic figures of every step (foreign-transcript removal, MetaSpots, gene embedding,
  LENS training, LENS against gene-shuffled cells, MetaDomains, annotation validation, label share against RNA depth,
  example cells). `intermediate_results/module5_lens_training.csv` holds the LENS training curve.
- `intermediate_results/module3_metaspots_per_cell_hist.png`: MetaSpots per cell after segmentation.

## Citation

Please cite the software as described in `CITATION.cff`.
