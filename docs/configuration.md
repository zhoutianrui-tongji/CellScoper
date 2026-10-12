# CellScoper Configuration Reference (params.yaml)

This document describes the configuration options read from `config/params.yaml`. All fields are optional unless stated otherwise. CLI flags control only high-level workflow; module hyperparameters are configured here.

- File path: `config/params.yaml`
- Format: YAML (UTF‑8)
- Scope: IO behavior, resume policy, module parameters (M1–M8), annotation, and UI/runtime hints

## Loading and precedence
- `--config` accepts YAML and is recursively merged over the packaged default.
- The merged configuration plus CLI overrides is atomically written to
  `effective_params.yaml`; all pipeline stages read that exact file.
- Explicit CLI/Python arguments take precedence over YAML. After a successful
  non-dry run, the same configuration is committed as `params_used.yaml`.

## io
- `final_format` (`csv|parquet`): Final table format. The shipped production
  configuration uses `parquet`; enable `write_csv_copy` only when a CSV is
  explicitly needed.
- `h5ad_compression` (`lzf|gzip|None`): Compression for `.h5ad` intermediates and outputs. Default: `lzf`.
- `lite_intermediate` (bool): Reduce intermediate artifacts written. Default: `true`.
- `intermediate_dirname` (str): Name of the intermediate directory under output dir. Default: `intermediate_results`.
- `resume_from` (`load_inputs|preprocess|build_anndata|annotate|write_outputs`|null): Force starting stage. Default: unset.
- `resume_policy` (`auto|minimal|force`): Default resume behavior (CLI can override). Default: `minimal`.
- `write_intermediate_csv` (bool): Write CSV intermediates when applicable. Default: `false`.
- `write_intermediate_parquet` (bool): Write Parquet intermediates when applicable. Default: `false`.
- `write_csv_copy` (bool): Also keep a CSV copy of the final table if feasible.
  The shipped production configuration defaults to `false` to avoid duplicating
  a gigabyte-scale table; set it to `true` when CSV is explicitly required.
- `write_diagnostic_tables` (bool): Write compact audit tables independently
  of large intermediate tables. Default: `false`.
- `write_diagnostic_csv`, `write_diagnostic_parquet` (bool): Formats for
  compact audit tables. Defaults: `false`, `false`.
- `broadcast_metaspot_diagnostics_to_transcripts` (bool): Duplicate wide
  MetaSpot audit columns onto every transcript. Default: `false`; MetaSpot
  diagnostics remain available in `adata1.obs`.
- `write_partitioned_dataset` (bool): Also write `final_data_dataset/` beside
  the canonical final table for streaming runs. Default: `false`.
- `drop_embeddings` (bool): Remove transcript embedding-like columns and write
  a compact final `adata1.h5ad`. The compact file retains the canonical
  `X_dgi_gcn_metaspot` representation needed for DGI provenance, but omits
  auxiliary embeddings and graph matrices. Default: `false`.
- `provenance_hash_inputs` (bool): Include exact SHA256 hashes of inputs and
  bundled annotation references in run manifests and resume fingerprints.
  Default: `true`.

## annotation

- `enable` (bool): Run subcellular annotation. Default: `true`.
- `method` (`evidence|none`): `evidence` is the only annotation method; `none`
  skips annotation. The earlier RAP-X method and its parameters were removed
  (last available at the git tag `legacy-rapx`); `method: rapx` is rejected
  with an error.

### Evidence annotation
A handful of parameters and a built-in check of its own labels. For each MetaSpot it compares observed marker transcripts with
what the owning cell's RNA mixture predicts, tests the excess against copies
of the data with gene identities shuffled within cells, and then validates each structure by holding marker
genes out (every gene exactly once) and asking whether they land in the
labelled regions: the typical held-out marker gene must be more enriched there
than the typical other gene.

Three label columns are written:

- `subcellular_annotation` is the final label. Inside each cell, every
  structure is placed on the unit with its highest score (never across the
  nuclear boundary), so nearly every cell shows all seven structures. It is not
  gated by validation.
- `subcellular_annotation_validated` keeps only structures that pass the
  held-out check in the current dataset, after adjusting for all
  structure/level combinations; other units are `Nucleus` or `Cytosol`.
- `subcellular_annotation_exploratory` adds structures passing without that
  adjustment.

Every structure keeps its `subcellular_score__*` / `subcellular_q__*` columns.
`evidence_annotation_report.json` states for each structure whether held-out
marker genes support it; `unsupported_structures` lists those whose final
labels are a best guess from the reference.

- `evidence_score` (`marker|propagated`): count reference markers only, or
  weight every gene by its within-cell co-localization with them. Default:
  `propagated`.
- `evidence_unit` (`auto|metaspot|metadomain`): unit of evidence. `auto` picks,
  per structure, the level with the stronger held-out enrichment.
- `evidence_fdr` (float): BH threshold for calling a unit. Raising it trades
  held-out enrichment for coverage; the report states both. Default: `0.05`.
- `evidence_validation_folds`, `evidence_null_sets`, `evidence_bootstrap`
  (int): validation folds, permutation replicates for the null, cell bootstraps.
  Defaults: `5`, `10`, `2000`.
- `evidence_min_gene_transcripts` (int): minimum transcripts per gene within a
  compartment. Default: `50`.
- `evidence_marker_margin` (float): lead a reference gene needs over its
  second-best compartment to count as a marker. Default: `0.2`.
- `evidence_min_resolution` (float): if the split-half Spearman correlation of
  gene nuclear preference is below this, the data are judged not to resolve
  subcellular localization and only Nucleus/Cytosol is reported. Default:
  `0.3`.

### Held-out reference validation
- `annotation.reference_exclude_genes_file` (path, optional): Text file with
  one gene symbol per line. Listed genes are removed from the APEX/RNALocate
  (or user-supplied) reference before annotation, so their localization can
  be used as an independent check of the resulting labels. See
  `benchmarks/evaluate_run.py`. Default: unset.

## module1 (Geometry features)
- `knn_k` (int): Neighborhood size for geometry context. Default: `32`.
- `include_directional` (bool): Add directional features (sin/cos). Default: `true`.
- `use_input_nucleus_flags` (bool): If `true`, reuse transcript-level
  `overlaps_nucleus`; if `false`, recompute nuclear membership from the
  supplied nucleus polygon. Benchmark configurations should normally use
  `false` to avoid treating a simulated truth column as a model input.
  Packaged default: `false`.
- `boundary_distance` (`vertex|edge`): How `d_cell`/`d_nuc` are measured.
  `vertex` (default, historical behavior) uses the nearest boundary vertex;
  `edge` uses the exact distance to the polygon edges, which is more accurate
  when boundary polygons have few vertices. Changing it reruns from Module 1.
- `remove_foreign_transcripts` (bool): Drop transcripts that probably belong to
  a neighbouring cell before any analysis. Cells are grouped into coarse types;
  a transcript is dropped when its gene is expressed in the cell's type at no
  more than `foreign_max_relative_expression` (default `0.2`) of the level in
  the type that expresses it most. Recommended for tissue sections, where
  mis-segmented neighbour transcripts form patches that look like subcellular
  structure. Default: `false`. Changing it reruns from preprocessing.
- `winsor_perc_low`/`winsor_perc_high` (int, 0–100): Winsorization percentiles. Default: `1`, `99`.

## module2 (Composition embeddings)
- `top_g` (int): Vocabulary size (top genes). Default: `1024`.
- `bow_knn_k` (int): Neighborhood size for BoW. Default: `12`.
- `max_rows_per_cell_for_svd` (int): Per-cell TF rows cap for SVD sample. Default: `256`.
- `max_svd_rows` (int): Global TF rows cap for SVD sample. Default: `100_000`.
- `svd_n_components` (int): Latent dimensionality. Default: `24`.
- `bow_weight_mode` (`gaussian|uniform`): Neighbor weighting. Default: `gaussian`.
- `use_tfidf` (bool), `tf_norm` (bool): SVD weighting and TF normalization. Default: `true`, `true`.
- `parallel_chunk_cells` (int): Grouping factor for parallel TF blocks. Default: `8`.
- `stream_local_embeddings` (bool): Compute transcript-level neighborhood expression inside Module 3 one cell at a time instead of storing dense embedding columns for all transcripts. Default: `true`.

## module3 (Per-cell MetaSpot segmentation)
- `parallel_chunk_cells` (int): Number of cells dispatched at a time through
  Module 3's persistent worker pool. Default: `256`; this controls scheduling
  overhead and does not change the requested MetaSpot count.
- `parallel_n_jobs` (int): Module 3 worker count, bounded by `runtime.n_jobs`
  and the process CPU affinity. Set `0` to use the conservative eight-worker
  automatic cap.
  Nested BLAS/OpenMP pools are limited to one thread during this block.
  Default: `12`.
- `hdbscan_min_cluster_size` (int): Minimum subcluster size. Default: `30`.
- `hdbscan_allow_single_cluster` (bool): Allow single-cluster HDBSCAN. Default: `true`.
- `hdbscan_min_samples_frac` (float): Fraction of min_cluster_size for `min_samples`. Default: `0.2`.
- `kmeans_min_k`/`kmeans_max_k` (int): Bounds for KMeans fallback. Default: `2`, `50`.
- `merge_min_size` (int): Merge clusters smaller than this threshold. Default: `20`.
- `enforce_spatial_connectivity` (bool): Build every final `metaspot` as a connected XY region using spatially constrained Ward clustering.
- `partition_by_nucleus` (bool): Allocate the cell-level MetaSpot budget
  across nucleus and cytoplasm before clustering. When enabled, a MetaSpot
  cannot cross the nuclear boundary. Packaged default: `true`; without it a
  boundary-straddling MetaSpot takes the majority compartment and mislabels
  the minority transcripts.
- `target_sites_per_metaspot` (int|null): Target number of unique XY sites per MetaSpot. When set, this determines the requested per-cell K before connected-component merging and bypasses HDBSCAN as the primary segmenter. Packaged default: `20`.
- `min_metaspots_per_cell` (int): Lower bound for the requested per-cell K. Packaged default: `10`.
- `max_metaspots_per_cell` (int|null): Upper bound for the per-cell K. Default: `100`.
- `spatial_connectivity_k` (int): Maximum nearby XY neighbors considered when constructing the local spatial graph. Default: `8`.
- `spatial_radius_quantile` (float): Quantile of per-site nearest-neighbor distances used to estimate native point spacing. Default: `0.9`.
- `spatial_radius_factor` (float): Multiplier applied to the estimated point spacing for the graph radius. Default: `1.5`.
- `spatial_smoothing_iterations` (int): Local majority-smoothing passes used only if constrained Ward clustering cannot run. Default: `1`.
- `spatial_smoothing_min_fraction` (float): Minimum weighted neighbor fraction required to change a site's preliminary label in that fallback. Default: `0.6`.
- `spatial_min_component_size` (int): Connected components smaller than this transcript count are merged across their strongest shared boundary. Default: `5`.
- Module 3 writes `module3_metaspot_segmentation_audit.csv` when compact
  diagnostics are enabled. It reports actual/requested MetaSpot counts, size
  distributions, and nuclear-mixing rates for every cell.

- `spatial_compactness` (float): Weight of standardized XY coordinates in the spatially constrained Ward clustering. Larger values make MetaSpots more compact; `0` uses features only while still restricting merges to spatial neighbors. Default: `1.0`.
- `hdbscan_*`, `target_sub_size`, and `kmeans_*`: Legacy fallback controls used when `target_sites_per_metaspot` is null or spatial connectivity is disabled.

## Fast exploratory overlay

`config/params.fast.yaml` is a partial configuration merged over the packaged
defaults by the CLI:

```bash
cellscoper ... --config config/params.fast.yaml
```

It opts into PyNNDescent for the exact-scale expression graph, caps CA-DMGI at
120 epochs, uses two consensus seeds, and disables UMAP/diagnostic figures.
This is intended for iteration and can change graph edges, embeddings, and
MetaDomains. The default `params.yaml` remains the production profile.

## module5 (Meta-domain clustering)
- `block_size` (int): Chunk size for processing. Default: `50000`.
- `clustering_method` (`consensus_leiden|minibatch_kmeans`): MetaDomain
  clustering backend. The packaged default is `consensus_leiden`: it builds a
  mutual-kNN graph in `X_dgi_gcn_metaspot`, runs every configured resolution
  and random seed, chooses the best stable candidate subject to size and
  cross-cell recurrence constraints, and consolidates assignments using
  sparse edge-level co-association. `minibatch_kmeans` is retained for legacy
  reproduction and ablation.
- `consensus_neighbors` (int): Neighbors used for the mutual-kNN graph.
  Default: `20`.
- `consensus_resolutions` (list[float]) and `consensus_seeds` (list[int]):
  resolution grid and independent Leiden seeds. Packaged defaults:
  `[1, 2, 4, 6, 8]` and `[17, 42, 89]`.
- `consensus_min_average_size`/`consensus_max_average_size` (float): feasible
  average MetaDomain-size interval. Packaged defaults: `100/800` MetaSpots.
- `consensus_min_recurrence_cells` (int),
  `consensus_max_small_cluster_fraction` (float), and
  `consensus_max_nonrecurrent_fraction` (float): reject resolutions dominated
  by tiny or single-cell-specific clusters. Defaults: `2`, `0.10`, `0.10`.
- `consensus_edge_agreement` (float): fraction of seed runs that must place
  both endpoints of a mutual-kNN edge in the same cluster before that edge is
  retained in the consensus graph. Default: `2/3`.
- `consensus_min_stability` (float) and `consensus_stability_tolerance`
  (float): retain the finest feasible candidate no more than `0.10` below the
  best feasible ARI. This avoids choosing a biologically uninformative coarse
  partition for a tiny numerical stability advantage. `consensus_min_stability`
  (`0.55`) is reported, not enforced: when the selected resolution falls below
  it, a warning is logged and `stability_floor_met` is `false` in the stored
  metadata. Using it as a hard gate made identical runs switch resolution,
  because seed-pair ARIs carry a few hundredths of noise.
- Consensus evidence is stored in
  `adata1.uns["module5_metadomain_clustering"]["consensus_by_block"]` and, when
  diagnostics are enabled, in `metadomain_consensus_leiden_diagnostics.png`
  plus `module5_consensus_leiden_diagnostics.*`.
- `w_gc`/`w_expr`/`w_sp` (float): Feature weights (geometry, expression, spatial). Packaged default: `0.4/0.4/0.2`.
- `use_gc`/`use_expr`/`use_spatial_feats`/`use_decell`/`use_cosine` (bool): Feature toggles. Defaults: `true/true/true/true/true` (spatial feats default may vary by build).
- `speed_preset` (`ultra|fast|balanced`): Internal speed/accuracy presets. Default: `ultra`.
- `enable_cap` (bool): Per-cell cap enabled. Default: `true`.
- `min_domain_size` (int): Clusters smaller than this are merged into the nearest sufficiently sized center after final assignment. Default: `5`.
- Legacy MiniBatchKMeans K estimation (used only when
  `clustering_method: minibatch_kmeans`):
  - `estimate_k` (`auto|sampled_knee|legacy_auto|fixed`): Strategy. `auto` and `sampled_knee` are aliases for bounded, cell-stratified MiniBatchKMeans search. `legacy_auto` restores the historical count-only heuristic. Default: `auto`.
  - `fixed_k` (int): Used when `estimate_k=fixed`. Default: `1024`.
  - `auto_k_min`/`auto_k_max` (int): Hard candidate bounds for sampled-knee selection. Defaults: `20/8000`.
  - `auto_k_min_average_size`/`auto_k_max_average_size` (int): Candidate K values are constrained so the expected average MetaDomain contains between these many MetaSpots. Defaults: `20/500`.
  - `auto_k_n_candidates` (int): Number of logarithmically spaced K candidates. Default: `8`.
  - `auto_k_train_sample`/`auto_k_validation_sample` (int): Maximum cell-stratified train and held-out validation sizes. Defaults: `50000/3000`.
  - `auto_k_sample_seed` (int), `auto_k_seeds` (list[int]): Reproducible sampling seed and MiniBatchKMeans stability seeds. Defaults: `2026` and `[42, 43]`.
  - `auto_k_stability_min` (float): Minimum held-out adjusted Rand agreement between seeds. If the distortion-knee candidate fails, selection backs off to a smaller feasible K. Default: `0.4`.
  - `auto_k_max_small_cluster_fraction` (float): Maximum sampled/full-evaluation point fraction assigned to clusters expected to fall below `min_domain_size`. Default: `0.01`.
  - `auto_k_min_train_per_cluster` (int): Candidate K cannot exceed the training sample size divided by this value. Default: `10`.
  - `auto_k_batch_size`/`auto_k_max_iter` (int): Search-only MiniBatchKMeans controls. Defaults: `2048/30`.
  - `auto_k_full_eval_max` (int): Candidate occupancy is evaluated on all MetaSpots up to this size and on the bounded sample above it. Default: `100000`.
  - `target_sub_per_cluster`, `target_cluster_per_cell`, and `K_min`: Legacy controls used only by `estimate_k=legacy_auto`. `K_max` remains a backward-compatible upper bound when `auto_k_max` is omitted.
  - The selected K and all candidate distortion/stability/occupancy diagnostics are written to `adata1.uns["module5_metadomain_clustering"]["k_selection"]`.
  - When intermediate figures are enabled, Module 5 writes `metadomain_k_selection_diagnostics.png` plus `module5_k_selection_diagnostics.*` with the plotted candidate-level evidence.
- Nucleus/cytoplasm split:
  - `split_by_nucleus` (bool), `nucleus_threshold` (float).
- Per-cell cap:
  - `cap_ratio_base` (float), `cap_min_keep` (int), `cap_margin_tau` (float), `cap_max_frac_soft` (float).

## module6 (Aggregation to adata2)
- `h5ad_compression_override` (bool): Respect IO compression override for `adata2`. Default: `true`.
- `save_shape_summary` (bool): Save `module6_adata2_shape.*`. Default: `true`.

Each run writes `OUTPUT_GUIDE.md` at the output root, listing the production
files, the annotation columns and the diagnostics that were produced.

Every CLI run writes `run_manifest.json` with status, effective configuration
hash, input/reference identity, source-tree identity, runtime overrides, and
actual output paths. Annotation failures are fatal when annotation is enabled;
the pipeline will not label an unannotated or stale result as successful.

## module7 (Graph construction and expression latent)
- `graph_unit` is an internal compatibility field derived from
  `module8.dgi_unit`. New configurations should leave it as `auto`.
- `scvi_epochs` (int), `scvi_n_latent` (int), `scvi_lr` (float), `scvi_batch_size` (int): SCVI training params.
- `rep_priority` (list[str]): Representation preference for annotation: `X_dgi_gcn`, `X_dgi_sage`, `X_expr`, `X_pca`.
- `neighbors.k_expr`/`neighbors.k_geom` (int): k for expression/geometry graphs. Default: `15/15`.
- `alpha_expr`/`alpha_geom` (float): Fusion weights. Default: `0.5/0.5`.
- `same_cell_beta` (float): Bias for same-cell edges. Default: `0.5`.
- `topk_prune` (int): Row-wise top-k pruning of edges. Default: `32`.
- `edge_norm` (`rw|gcn|none`): Edge normalization. Default: `rw`.
- `feature_scale` (`standard|none`): Feature scaling mode. Default: `standard`.
- `knn_backend` (`auto|exact|pynndescent|faiss_hnsw|faiss_ivfpq`): kNN
  backend selected by the scale-aware execution plan. In the exact/full-batch
  profile, `auto` keeps the historical exact sklearn graph; an explicit
  `pynndescent` or `faiss_hnsw` opts into approximate graph membership while
  retaining full-batch DGI. `faiss_ivfpq` remains streaming-only.
- In MetaSpot mode, `neighbors.k_expr` and `neighbors.k_spatial` control two
  separately retained relation graphs:
  `obsp["metaspot_expr_connectivities"]` and
  `obsp["metaspot_spatial_connectivities"]`.
- `metaspot_expr_cross_cell_only` (bool): remove same-cell edges from the
  expression relation after expression-neighbor construction. The packaged
  CA-DMGI configuration sets this to `true`, so cross-cell reproducibility and
  within-cell spatial continuity cannot collapse into the same edge type.
- `metaspot_expr_n_components` (int): expression SVD/PCA dimension used for
  expression-neighbor discovery. Packaged default: `32`.
- `metaspot_use_geometry_features` (bool): append robust MetaSpot geometry to
  node features. Relation membership still comes from expression or spatial
  adjacency; geometry is not used as an annotation label.

## module8 (DGI embeddings)
- `dgi_unit` (`metadomain|metaspot`): Single switch controlling the DGI
  observation unit and module order. Packaged default: `metaspot`.
  - `metadomain`: run M5/M6 first, then train DGI on `adata2`. This is the
    scalable alternative for large datasets.
  - `metaspot`: train DGI on `adata1`, use
    `X_dgi_gcn_metaspot` for M5 clustering, and pool it into `adata2`.
    `module7.max_full_batch_nodes` is a hard safety limit; the current
    implementation does not silently fall back or pretend that an oversized
    MetaSpot graph completed. Its packaged default is `200000`; raising it
    only disables the guard and does not reduce full-batch memory cost.
- `dgi_framework` (`cell_aware_dmgi|standard_dgi`): training objective. The
  packaged default, `cell_aware_dmgi`, uses two relation-specific residual GCN
  encoders: the cross-cell expression graph and the within-cell spatial graph.
  Each branch optimizes a local Infomax discriminator against features
  permuted within the same owning cell. Thus no real MetaSpot is declared a
  biological negative. A bounded node-wise gate combines the two embeddings
  into the public compatibility key `X_dgi_gcn_metaspot`; branch embeddings
  are retained as `X_dmgi_expression_metaspot` and
  `X_dmgi_spatial_metaspot`. `standard_dgi` retains the historical single-graph
  implementation for ablation and old configurations.
- `dmgi_expression_graph_key`/`dmgi_spatial_graph_key` (str): explicit
  relation-graph keys. Defaults: `metaspot_expr_connectivities` and
  `metaspot_spatial_connectivities`.
- `dmgi_expression_weight`/`dmgi_spatial_weight` (float): relation-specific
  Infomax weights. Both default to `1.0`.
- `dmgi_consensus_weight` (float): alignment weight between the gated
  consensus and both relation embeddings. Default: `0.1`.
- `dmgi_gate_floor` (float): lower bound for either relation's node-wise gate;
  `0.1` constrains expression and spatial contributions to `[0.1, 0.9]` and
  prevents silent branch collapse.
- `dmgi_fused_corruption_pass` (bool): propagate the positive and within-cell
  corrupted views together so each relation/layer scans its sparse edges once
  instead of twice. Default: `true`. With dropout disabled this is numerically
  identical to the separate-pass implementation; with dropout enabled it
  preserves the same objective and independent dropout sampling.
- `dmgi_early_stopping`, `dmgi_min_epochs`, `dmgi_patience`, and
  `dmgi_min_delta`: EMA-based stopping controls. The packaged configuration
  uses `true`, `40`, `15`, and `1e-4`; `dgi_epochs: 200` is the maximum rather
  than a mandatory run length.
- `dgi_epochs` (int): Maximum training epochs. Packaged default: `200`.
- `enable_dgi_sage` (bool): Train SAGE variant in addition to GCN. Default: `false`.
- `dgi_encoder` (`residual|plain`): Encoder architecture. `residual` uses an
  input projection, initial-feature injection, residual graph-convolution
  blocks, and optional LayerNorm. `plain` preserves the historical sequential
  GCN. Default: `residual`.
- `dgi_layers` (int): Number of graph message-passing layers. This is also the
  graph receptive-field depth; increasing it is not equivalent to adding only
  feature-transform capacity. Default: `2`.
- `dgi_hidden`/`dgi_out` (int): Hidden and output dimensions. Default:
  `64/32`. To reproduce embeddings produced before the residual-encoder
  update, set `dgi_encoder: plain`, `dgi_layers: 2`, `dgi_dropout: 0.0`, and
  `dgi_out: 64` together.
- `dgi_dropout` (float): Dropout used by the encoder. Default: `0.1`.
- `dgi_residual_alpha` (float): Fraction of the initial projected features
  injected into every residual graph block. Default: `0.1`.
- `dgi_layer_norm` (bool): Apply LayerNorm after residual graph blocks.
  Default: `true`.
- `dgi_device` (`auto|cpu|cuda`): execution device. `auto` selects CUDA only
  when the installed PyTorch build and driver expose an available device.
  Explicit `cuda` fails fast instead of silently running on CPU.
- `dgi_cpu_threads` (int): PyTorch intra-op threads used by DGI on CPU. `0`
  auto-selects up to 32 CPUs from the process affinity; positive values are
  explicit overrides. This is independent from preprocessing `blas_threads`.
- `dgi_sparse_layout` (`auto|csr|coo`): sparse relation-graph storage for
  CA-DMGI. `auto` uses CSR with compact native SciPy indices; `coo` is a
  compatibility fallback.
- `dgi_tf32` (bool): allow TF32 dense matrix multiplication on supported CUDA
  hardware. Sparse graph values and model outputs remain float32.
- `dgi_corruption_mode` (`auto|global|same_cell`): Negative-view feature
  corruption. In MetaSpot mode, `auto` uses a non-self permutation within each
  `cell_id`, so the discriminator must learn subcellular organization rather
  than relying on easy between-cell expression differences. In MetaDomain
  mode, `auto` uses the historical global permutation. This switch belongs to
  the `standard_dgi` fallback; CA-DMGI always records same-cell corruption.
- `dgi_local_loss_weight` (float): Weight of an unsupervised local edge
  contrastive loss added to standard DGI. Positive pairs are graph edges;
  negative pairs are non-edges from the same cell when `cell_id` is available.
  Packaged default: `0.0`, because a same-cell non-edge may still represent the
  same biological structure. Set it explicitly only for the legacy ablation.
- `dgi_local_temperature` (float): Cosine-logit temperature for the local
  contrastive term. Default: `0.2`.
- `dgi_local_samples` (int): Maximum positive edges sampled once per training
  run for the local term. Default: `65536`.
- `dgi_local_graph_key` (`auto` or an `obsp` key): With `auto`, prefer
  `metaspot_spatial_connectivities` and otherwise use the main DGI graph.
  Default: `auto`.
- `dgi_cpu_backend` (`auto|torch_csr|pyg`): CPU GCN execution backend. `auto`
  uses the equivalent pre-normalized PyTorch CSR SpMM path on CPU and retains
  PyG on GPU for `standard_dgi`; `pyg` restores the legacy COO scatter
  implementation. CA-DMGI uses `dgi_sparse_layout` instead.
- `dgi_lr` (float), `dgi_weight_decay` (float), `dgi_clip_grad` (float): Optimizer settings.

The packaged configuration deliberately keeps two propagation layers. CA-DMGI
is fully unsupervised: it uses expression, geometry, cell ownership and spatial
adjacency, but never final subcellular labels. Model selection should therefore
use cross-seed/cross-cell clustering stability, representation ablations and
downstream evidence support—not visual UMAP separation or training loss alone.
`module8_ca_dmgi_training.png` reports both relation losses and the gate
trajectory; exact histories, graph keys, corruption semantics and gate metrics
are stored under `adata1.uns["dgi_embeddings"]` and exported to
`module8_ca_dmgi_training_history.*`/`module8_ca_dmgi_summary.*`. To reproduce
the historical single-graph objective, set `dgi_framework: standard_dgi`,
`dgi_corruption_mode: global`, and `dgi_local_loss_weight: 0.0`.

`module5.cluster_rep`, `module6.pool_reps`, and `module7.graph_unit` are
materialized automatically in `effective_params.yaml`. Legacy configurations
that only set `module7.graph_unit` remain supported.

## runtime (parallelism)
- `n_jobs` (int): Preferred worker count for parallel sections (honored unless `CELLSCOPE_N_JOBS` env is set).
- `joblib_backend` (`threading|loky`): Backend for joblib blocks (env `CELLSCOPE_JOBLIB_BACKEND` still wins).
- `blas_threads` (int): Caps BLAS/OpenMP thread pools; applied at startup and optionally via `threadpoolctl`. CellScoper forces this to `1` when a multi-worker `threading` backend is active, because OpenBLAS is not safe under nested application threading.
- `numba_threads` (int): Import-time Numba/PyNNDescent thread ceiling. With a multi-worker `threading` backend, CellScoper caps this so `n_jobs * numba_threads` does not exceed the bundled OpenBLAS thread-metadata limit.
- `scanpy_n_jobs` (int): Scanpy neighbor-graph workers, capped at the resolved `numba_threads` value. `SCANPY_N_JOBS` remains an environment override but is subject to the same cap.
- `use_threadpoolctl` (bool): If true and `blas_threads` is set, also call `threadpoolctl.threadpool_limits`.
- `neighbors_n_jobs` (int): Reserved for neighbor graph overrides (falls back to `n_jobs`).

## ui
- `show_sample` (bool): Show top-5 preview of inputs at start. Default: `true`.
- `progress_style` (`rich|plain`): Output style hint for CLI progress.

## Environment variables
- `CELLSCOPE_N_JOBS` (int): Override parallel job count for selected blocks.
- `CELLSCOPE_CLUSTER_CHUNK` (int): Chunk size for clustering batches (default ~32–128).

## Examples

Minimal (CSV output, defaults):
```yaml
io:
  final_format: csv
annotation:
  enable: true
```

Parquet output + strong pruning + faster DGI:
```yaml
io:
  final_format: parquet
  h5ad_compression: lzf
  lite_intermediate: true
module7:
  topk_prune: 16
module8:
  dgi_unit: metadomain
  dgi_epochs: 60
  enable_dgi_sage: false
```

MetaSpot-first DGI for a dataset below the configured full-batch node limit:
```yaml
module8:
  dgi_unit: metaspot
module7:
  max_full_batch_nodes: 200000
```

Resume controls in YAML (can also use CLI):
```yaml
io:
  resume_policy: minimal
  resume_from: build_anndata
```

## Change impact and resume
- The pipeline fingerprints module-relevant parts of the YAML. With `--resume-policy minimal`, it reruns from the earliest affected stage.
- First full run prints `Module X · Done` (no suffix). Subsequent runs show `(cache|recomputed)`.

## Troubleshooting
- Changes not taking effect: ensure `resume_policy` is `minimal` or use `--resume-policy minimal` and/or remove the output directory.
- Missing dependencies (e.g., `scanpy`/`scvi-tools`): the pipeline will use fallback paths (e.g., SVD/NN); adjust epochs or disable annotation if needed.
- Large runs: use `module8.dgi_unit: metadomain` when the expected number of
  MetaSpots exceeds `module7.max_full_batch_nodes`. Lowering epochs does not
  remove the full adjacency/feature memory requirement. MetaSpot-DGI above the
  limit requires a sampled or mini-batch graph implementation.
