# CellScoper Configuration Reference (params.yaml)

This document describes the options read from `config/params.yaml`. All fields are optional; omitted fields take the
defaults listed here. CLI flags control the workflow (inputs, output directory, resume); module parameters are set in
YAML.

- File path: `config/params.yaml`
- Format: YAML (UTF-8)
- Sections: `io`, `execution`, `module1`-`module3`, `module5`, `module6`, `annotation`, `runtime`

## Loading and precedence
- `--config` accepts a YAML file that is recursively merged over the packaged default; it only needs the keys it changes.
- The merged configuration plus CLI overrides is written atomically to `effective_params.yaml`; every stage reads that
  file (also through the `CELLSCOPER_CONFIG` environment variable). After a successful run the same configuration is
  committed as `params_used.yaml`.
- Explicit CLI/Python arguments take precedence over YAML.

## io
- `final_format` (`csv|parquet`): final transcript table format. Default: `parquet`.
- `h5ad_compression` (`lzf|gzip|None`): compression of `.h5ad` files. Default: `lzf`.
- `write_csv_copy` (bool): also write a CSV copy of the final table. Default: `false`.
- `cell_structure_features` (bool): write `cell_structure_features.parquet` (needs annotation). Default: `true`.
  See "Cell x structure table" below.
- `write_intermediate` (bool): write checkpoints under `intermediate_dirname`. Default: `true`.
- `lite_intermediate` (bool): keep only the checkpoints needed for resume. Default: `true`.
- `intermediate_dirname` (str): checkpoint directory under the output directory. Default: `intermediate_results`.
- `write_intermediate_csv`, `write_intermediate_parquet` (bool): formats of intermediate tables. Default: `false`.
- `write_intermediate_figures` (bool): write diagnostic figures. Default: `true`.
- `write_diagnostic_tables`, `write_diagnostic_csv`, `write_diagnostic_parquet` (bool): compact audit tables such as
  the Module 3 segmentation audit. Default: `false`.
- `write_partitioned_dataset` (bool): for `streaming` runs, also write `final_data_dataset/`, a cell-aligned partitioned
  Parquet dataset with `manifest.json` and `_SUCCESS`. Default: `false`.
- `drop_embeddings` (bool): write a compact `adata1.h5ad` that keeps only `X_lens` of the MetaSpot
  representations. Default: `false`.
- `provenance_hash_inputs` (bool): include SHA256 hashes of the inputs and bundled references in `run_manifest.json`
  and the resume fingerprints. Default: `true`.
- `resume_policy` (`auto|minimal|force`) and `resume_from`
  (`load_inputs|preprocess|build_anndata|annotate|write_outputs`|null): see "Change impact and resume".
- `show_sample` (bool, top level): print the first rows of each input. Default: `true`.

## execution
`profile: auto` routes from the realized MetaSpot count: `exact` up to `exact_max_metaspots` (200,000), `sampled` up to
`sampled_max_metaspots` (2,000,000), `streaming` above (full Atera has ~32M MetaSpots). `streaming` skips full diagnostic
tables and enables the partitioned final dataset (`io.write_partitioned_dataset`, `partition_rows` rows per part,
default 1,000,000). The plan is written to `intermediate_results/execution_plan.json` and `adata1.uns["execution_plan"]`.

## module1 (geometry)
- `include_directional` (bool): add directional features (sin/cos of the angle around the nucleus). Default: `true`.
- `use_input_nucleus_flags` (bool): `true` reuses the transcripts' `overlaps_nucleus`; `false` recomputes nuclear
  membership from the nucleus polygons. Default: `false`.
- `boundary_distance` (`vertex|edge`): `d_cell` / `d_nuc` as distance to the nearest boundary vertex (default) or to the
  polygon edges (more accurate for polygons with few vertices).
- `remove_foreign_transcripts` (bool) and `foreign_max_relative_expression` (float): drop transcripts that probably
  belong to a neighbouring cell. Cells are grouped into coarse types; a transcript is dropped when its gene is expressed
  in the cell's type at no more than this fraction (0.2) of the level in the type that expresses it most. Segmentation
  spillover forms patches that look like subcellular structure (about 5-6% of transcripts in the tissue sections
  tested). Default: `true`.
- `winsor_perc_low` / `winsor_perc_high` (int): winsorization percentiles of the geometry features. Default: `1`, `99`.

## module2 (local composition)
Each transcript is described by the distance-weighted bag of the `top_g` (1024) most frequent genes among its
`bow_knn_k` (12) nearest transcripts of the same cell, with TF normalisation (`tf_norm`) and IDF weighting
(`use_tfidf`, `idf_mode`: `log1p|log|none`), projected to `svd_n_components` (24) dimensions by a sparse random
projection. `bow_weight_mode` (`gaussian|inv|uniform`) weights the neighbours. The IDF is estimated on a sample of at
most `svd_sample_cells` (5000) cells, `max_rows_per_cell_for_svd` (256) transcripts per cell and `max_svd_rows`
(100,000) rows. The model is applied cell by cell inside Module 3, so no per-transcript embedding table is stored.
`parallel_chunk_cells` (8) sets the cells per parallel task.

## module3 (MetaSpots)
Every cell side (nucleus, cytoplasm) is split into spatially connected MetaSpots by spatially constrained Ward
clustering of its transcripts on standardised geometry + Module 2 features; MetaSpots never cross the nuclear boundary
(2k-cell Xenium audit: nucleus/cytoplasm accuracy 0.896 -> 0.951 against mixed MetaSpots).
- `target_sites_per_metaspot` (int): about one MetaSpot per this many distinct x-y sites. Default: `15`.
- `min_metaspots_per_cell` / `max_metaspots_per_cell` (int): bounds of the per-cell MetaSpot count. Default: `20`, `1000`.
- `use_z_location` (bool): with `z_location`, connectivity and clustering in 3D (one length scale); the MetaSpot count
  still comes from the x-y sites. Default: `true`.
- `n_segmentation_features` (int): the Ward features are the ones with the largest variance in the cell. Default: `24`.
- `min_segment_points` (int): a cell side with fewer transcripts forms one MetaSpot; also the smallest component when
  `spatial_min_component_size` is unset. Default: `30`.
- `spatial_connectivity_k`, `spatial_radius_quantile`, `spatial_radius_factor` (8, 0.9, 1.5): the spatial graph links
  each site to at most k neighbours within factor x the quantile of the nearest-neighbour spacing.
- `spatial_min_component_size` (int): smaller connected pieces merge into a neighbour. Default: `5`.
- `spatial_compactness` (float): weight of the coordinates in the Ward features. Default: `1.0`.
- `spatial_smoothing_iterations`, `spatial_smoothing_min_fraction` (1, 0.6): majority smoothing used only when
  constrained Ward cannot run.
- `parallel_chunk_cells` (256), `parallel_n_jobs` (12), `small_backend_cells` (32), `sequential_cells` (32): scheduling
  (cells per task, workers, thresholds for threads / sequential processing); they do not change the result.
- With compact diagnostics enabled, `module3_metaspot_segmentation_audit.*` reports actual and requested MetaSpot
  counts, sizes and nuclear mixing for every cell.

## module5 (MetaSpot representation and MetaDomains)

### MetaSpot representation
Two parts, both stored in `adata1.obsm`: the spatial gene embedding (`X_gene_embedding`) and LENS (`X_lens`).
Annotation reads both along each structure's marker direction; MetaDomains use LENS.

**Spatial gene embedding** (`cellscoper/spatial_gene_embedding.py`): each gene gets a vector from within-cell gene-pair
co-localisation; a MetaSpot is the mean of its transcripts' neighbourhood vectors.
- `gene_embedding_scale` (`multiscale|density`): `multiscale` (default) concatenates embeddings at 3, 7 and 14 x the
  median nearest-neighbour distance; `density` uses one scale.
- `gene_embedding_pair_radius` / `gene_embedding_neighbour_radius` (`auto` or a number in coordinate units).
- `gene_embedding_dim` (32 per scale), `gene_embedding_min_gene_count` (100), `gene_embedding_pseudo_count` (5.0),
  `gene_embedding_max_cells` (20,000 cells used to fit the gene vectors).

**LENS** (`cellscoper/lens.py`): a two-layer Transformer reads a MetaSpot's transcripts (frozen GE gene tokens
plus positions) and up to 32 nearby transcripts of the same cell side, and outputs a 32-d z. Training hides 25% of the
MetaSpot's genes and predicts them against the cell side's own composition, so z describes where the MetaSpot departs
from its cell side; genes still visible in the input are excluded from the answer. 10% of cells are held out for early
stopping. z is centred per cell side. A CUDA GPU is used when available (the one with the most free memory).
- `lens_max_epochs` (200): maximum epochs (Atera 5k stops after about 70-80).
- `lens_spots_per_epoch` (0): > 0 trains each epoch on that many random MetaSpots, so training time does not grow with
  the data (use for full-size datasets); early stopping then counts MetaSpots seen.
- `lens_null_cells` (10,000): after training, cells with gene identities shuffled within each cell side are encoded once;
  annotation and the MetaDomains use them as the null.
- `lens_background` (`own|learned`, default `own` since v1.0.0): the composition the hidden genes are predicted against.
  `own` = the cell side's own RNA without the MetaSpot (shrunk towards the cell's low-rank profile), so LENS learns only
  where a MetaSpot departs from its own cell, not cell-type composition; required by `evidence_label_rule: lens_count`.
  `learned` = a learned mix of the cell's own composition, a low-rank profile and the whole-section composition.
- `lens_absolute_position` (bool, default `false` since v1.0.0): also give LENS each transcript's radial position
  (relative distances to the nucleus and the cell edge). With the `own` background the radial position took over z in
  simulations; without it, only the nucleus/cytoplasm flag and positions relative to the MetaSpot are used.

### MetaDomains
MetaDomains are structure types found from LENS without reference genes (`cellscoper/metadomain_discovery.py`; until
2026-11-03 they were 25 MiniBatchKMeans clusters per side). `random_seed` (42) seeds LENS and the discovery.
1. z is smoothed over about 50 transcripts of neighbouring MetaSpots of the same cell side and whitened by the null's
   covariance (`lens_null_cells`).
2. MetaSpots whose whitened norm beats the null at false share `metadomain_q` (0.2) are detected, with
   thresholds per side x depth quartile.
3. HDBSCAN on the detected MetaSpots' unit directions (10 PCA dimensions, at most 20,000 MetaSpots, the same number
   from each side) gives candidate directions; the number comes from the data.
4. A candidate is kept when both random halves of the cells find it (mutual best, cosine >= 0.8).
5. Steps 3-4 are repeated 5 times with different samples and cell halves; directions kept in at least 3 of the 5
   restarts (grouped at cosine >= 0.8) become MetaDomains. A single restart depends on its sample: on Atera 5k the same z
   gave 0-4 MetaDomains across discovery seeds; with the restarts, both LENS seeds and three discovery seeds each gave
   the same 4 MetaDomains (decoder profile r >= .999 for the same z, .87-.94 between LENS seeds; about 80 s).
6. Every kept direction becomes an axis with its own null threshold; a MetaSpot takes the MetaDomain whose threshold it
   exceeds most, else stays background (-1).

Nucleus and cytoplasm are clustered together: clustered per side, Atera 5k gave mirror MetaDomains on both sides of the
2D nucleus outline, where RNA composition hardly changes. A nucleus-boundary check flags a MetaDomain whose genes follow
the raw nucleus-cytoplasm contrast (|r| >= 0.5) while >= 20% of its MetaSpots lie on the other side of the outline; it
reflects the outline (2D projection, segmentation) or a perinuclear gradient.

Results: `adata1.obs["meta-domain"]` (-1 = background) and `"meta-domain_score"`, the `meta-domain` column of
`final_data`, `adata2` (one row per MetaDomain and one for the background), and `adata1.uns["metadomain_discovery"]` (side, nucleus share, size, reproducibility, support over restarts, null share,
nucleus-boundary check, decoder gene profiles and top genes). Gene-shuffled input gives no MetaDomain on Atera, HEK and
CosMx; MetaDomains found with separately trained models matched for 78-84% (Atera) and 100% (HEK) of their MetaSpots.

## module6 (MetaDomain AnnData)
- `h5ad_compression_override` (bool): apply `io.h5ad_compression` to `adata2`. Default: `true`.
- `save_shape_summary` (bool): save `module6_adata2_shape.*`. Default: `true`.

## annotation
- `enable` (bool): run subcellular annotation. Default: `true`.
- `method` (`evidence|none`): `none` skips annotation.

### Label rule
`evidence_label_rule` (`lens_count|axis`, default `lens_count` since v1.0.0). `lens_count` (`cellscoper/lens_count.py`):
a structure is labelled only if held-out reference markers rank above expression-matched random gene sets in the LENS
decoder (z >= 3); the number of transcripts labelled per cell side comes from LENS gene weights along the marker
direction (one Otsu threshold per structure over all MetaSpots of its side, so amounts are comparable across cells);
where they go within the cell follows the mean of the evidence rank and the structure-axis score rank below. `axis`:
labels are the MetaSpots above the structure-axis threshold described below.

### Structure axes
Reference markers only name a direction: the mean vector of a structure's markers minus the mean vector of all genes,
in GE (per scale, whitened by the null) and in the LENS decoder. A MetaSpot's score is its representation on that
direction, averaged over about `evidence_axis_smooth_transcripts` (50) transcripts of neighbouring MetaSpots of its cell
side. Gene identities are shuffled within cell sides and the score is recomputed; a structure takes the MetaSpots above
the lowest score where the null's share above it is at most `evidence_axis_q` (0.2) times the real share. A structure
that never reaches this gets no MetaSpot. `evidence_axis_source` (`auto|ge|lens|ge+lens`; `auto` = ge+lens when LENS
ran) picks the read-out. Same-area held-out marker enrichment (OR at 5% area): Atera 1k 1.53 vs 1.29 with the earlier
count-based rule, HEK 1.22 vs 1.16, CosMx 1.21 vs 0.99, Atera 5k 1.67 vs 1.42; on gene-shuffled input it labels 0-0.9%
of MetaSpots against 3.8-10.5%.

Label columns: `subcellular_annotation` (final label), `subcellular_annotation_support` (`segmentation`, `validated`,
`unvalidated`), `subcellular_annotation_validated` (only structures that passed the held-out check),
`subcellular_annotation_exploratory` (relaxed check), `subcellular_annotation_placed` (each structure's best MetaSpot per
cell, display only), `subcellular_score__*` and `subcellular_q__*` (estimated false-label share at that score).

### Held-out validation (reported, does not change labels)
Every marker gene is held out once (`evidence_validation_folds`, 5): it is then neither a neighbour in the MetaSpot
vectors nor part of a marker direction, and its enrichment in the labelled MetaSpots is compared with other genes
(`evidence_bootstrap` 2000 resamples, `evidence_axis_validation_cells` 3000 cells, genes with at least
`evidence_min_gene_transcripts` 50 transcripts). `evidence_max_cells` (10,000) bounds the cells used to set thresholds;
scores and labels cover every cell. `evidence_marker_margin` (0.2) is the lead a reference gene needs over its second
structure to count as a marker. `evidence_min_resolution` (0.3): if the split-half Spearman correlation of gene nuclear
preference is below it, only Nucleus/Cytosol is reported. `evidence_seed` (0).

### Reference
Without `--gold-data`, the reference is the bundled APEX-seq atlas (Fazal et al. 2019) merged with RNALocate v3, plus
`src/extra_reference_markers.tsv`:

| Structure | Source | Genes |
|---|---|---|
| ER | CAP-seq ER-membrane RNAs (Wang et al. 2019, Nat Chem Biol 15:1110) | 329 |
| OMM | LoRNA mitochondrion fraction (Villanueva et al. 2024, Nat Methods 21:60) | 236 |
| Nuclear Speckle | APEX-seq of SRSF1/SRSF7/RNPS1 (Barutcu et al. 2022, Mol Cell 82:1035) | 289 |

When the data measure at least four mitochondrially encoded RNAs (`MT-*`), they replace the OMM markers. A structure
with no marker gene in the panel is not labelled and is listed under `structures_without_markers`.
`reference_exclude_genes_file` (path) removes listed genes from the reference, e.g. to keep them for an independent check.
A user-supplied `--gold-data` table is used as given.

## Cell x structure table
`io.cell_structure_features: true` writes `cell_structure_features.parquet` (`cellscoper/cell_features.py`): one row per
cell side and structure label (and MetaDomain) with
- `metaspot_share`, `transcript_share`: share of the side's MetaSpots / transcripts carrying the label (amount; rises
  with transcript depth, so compare cell types at matched depth);
- `pairs`, `pair_obs`, `pair_exp`, `concentration`: neighbour pairs of different genes within the middle GE radius,
  weighted by each gene's non-negative loading on the structure's GE direction (MetaDomains: the positive part of their
  decoder profile), over the expectation with gene names shuffled in the cell side. The ratio has no depth bias in
  expectation but is noisy per cell; compare groups with `pooled_concentration` (sum `pair_obs` / sum `pair_exp`). It
  measures arrangement: a structure that fills the cell (ER in plasma cells, Atera 5k) has a large share and a ratio
  near 1.

Atera 5k: 38 s with 32 jobs for 5000 cells; depth-matched ER share was higher in plasma cells than in T cells
(+0.098, 95% CI 0.061-0.144).

## runtime (parallelism)
- `n_jobs` (int): worker count for parallel sections (`CELLSCOPER_N_JOBS` overrides).
- `joblib_backend` (`threading|loky`): joblib backend (`CELLSCOPER_JOBLIB_BACKEND` overrides).
- `blas_threads` (int): BLAS/OpenMP thread cap, set before numpy is imported; forced to 1 with a multi-worker
  `threading` backend, because OpenBLAS is not safe under nested application threads.
- `numba_threads` (int): import-time Numba thread ceiling. `scanpy_n_jobs` (int): value of `SCANPY_N_JOBS` set at
  import, for downstream analysis in the same process (the pipeline does not use scanpy).
- `use_threadpoolctl` (bool): also apply `blas_threads` through `threadpoolctl`.

## Environment variables
- `CELLSCOPER_CONFIG`: effective configuration file (set by the CLI).
- `CELLSCOPER_N_JOBS`, `CELLSCOPER_JOBLIB_BACKEND`: override `runtime.n_jobs` / `runtime.joblib_backend`.
- `CELLSCOPER_CLUSTER_CHUNK`: cells per Module 3 batch.
- `CELLSCOPER_FORCE_THREADS`: run Module 3 with threads instead of processes.

## Examples

CSV output with defaults:
```yaml
io:
  final_format: csv
```

Finer MetaSpots (as in `config/params.atera_brca.yaml`):
```yaml
module3:
  target_sites_per_metaspot: 6
```

Full-size dataset (LENS trained on a fixed number of MetaSpots per epoch, partitioned final table):
```yaml
module5:
  lens_spots_per_epoch: 1000000
io:
  write_partitioned_dataset: true
```

## Change impact and resume
The pipeline fingerprints the configuration keys of each stage and module (`cellscoper/config.py`). With
`resume_policy: minimal` it reruns from the earliest stage whose keys, inputs or schema changed; `force` reruns
everything; `auto` continues after the last completed stage. `resume_from` starts from a named stage. The first full run
prints `Module X · Done`; later runs add `(cache)` or `(recomputed)`. `--dry-run-diff` shows which stage would rerun.

Every CLI run writes `run_manifest.json` with status, effective configuration hash, input and reference identity,
source-tree identity and output paths. Annotation failures are fatal when annotation is enabled.
