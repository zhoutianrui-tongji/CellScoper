import os
import copy
import yaml
import json
import hashlib
from typing import Any, Dict, Mapping, Optional, Tuple

from .resource_paths import default_config_path


BUILD_PIPELINE_SCHEMA_VERSION = 9
PREPROCESS_SCHEMA_VERSION = 2
ANNOTATION_SCHEMA_VERSION = 16
MODULE_SCHEMA_VERSIONS: Dict[str, int] = {
    'm1': 2,
    'm2': 2,
    'm3': 4,
    'm5': 9,
    'm6': 4,
}

_ANNOTATION_INTERNAL_DEFAULTS: Dict[str, Any] = {
    "enable": True,
    "method": "evidence",
}


def resolve_annotation_configuration(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the annotation section (or a bare annotation mapping) with defaults."""
    if not isinstance(config, Mapping):
        raw: Mapping[str, Any] = {}
    elif "annotation" in config:
        raw = config.get("annotation", {}) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("annotation must be a mapping")
    elif any(
        key in config
        for key in (
            "io", "module1", "module2", "module3", "module5", "module6",
            "runtime", "run",
        )
    ):
        raw = {}
    else:
        raw = config
    resolved = copy.deepcopy(_ANNOTATION_INTERNAL_DEFAULTS)
    resolved.update(copy.deepcopy(dict(raw)))
    return resolved


def _params_path() -> str:
    return default_config_path()


def get_default_params_path() -> str:
    return _params_path()


def load_params_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    """Load params YAML.

    If ``path`` is provided, load from that file; otherwise fall back to the
    ``CELLSCOPER_CONFIG`` env var (if set), then repo default ``config/params.yaml``.
    """
    env_p = os.environ.get("CELLSCOPER_CONFIG", "").strip()
    explicit = bool(path or env_p)
    if path:
        p = path
    else:
        p = env_p if env_p else _params_path()
    try:
        if not os.path.isfile(p):
            if explicit:
                raise ValueError(f"configuration file does not exist: {p}")
            return {}
        with open(p, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
    except ValueError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        if explicit:
            raise ValueError(f"could not parse configuration {p}: {exc}") from exc
        return {}
    if not isinstance(loaded, Mapping):
        if explicit:
            raise ValueError(
                f"configuration root must be a mapping, got {type(loaded).__name__}"
            )
        return {}
    return copy.deepcopy(dict(loaded))


# Map config keys to pipeline stages for minimal re-run decisions.
# Only names that appear in `pipeline.stage_order` are relevant here.
_BUILD_STAGE_KEYS: Tuple[str, ...] = (
    # Scale-aware execution
    'execution.profile','execution.exact_max_metaspots','execution.sampled_max_metaspots','execution.partition_rows',
    # Module 1
    'module1.include_directional','module1.use_input_nucleus_flags','module1.winsor_perc_low','module1.winsor_perc_high','module1.boundary_distance','module1.remove_foreign_transcripts','module1.foreign_max_relative_expression',
    # Module 2
    'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.idf_mode','module2.parallel_chunk_cells','module2.svd_sample_cells',
    # Module 3
    'module3.min_segment_points','module3.n_segmentation_features','module3.spatial_connectivity_k','module3.spatial_radius_quantile','module3.spatial_radius_factor','module3.spatial_smoothing_iterations','module3.spatial_smoothing_min_fraction','module3.spatial_min_component_size','module3.spatial_compactness','module3.target_sites_per_metaspot','module3.min_metaspots_per_cell','module3.max_metaspots_per_cell','module3.use_z_location','module3.segmentation_geometry','module3.segmentation_spatial_share',
    # Module 5
    'module5.random_seed','module5.metadomain_q','module5.gene_embedding_pair_radius','module5.gene_embedding_neighbour_radius','module5.gene_embedding_scale','module5.gene_embedding_dim','module5.gene_embedding_min_gene_count','module5.gene_embedding_pseudo_count','module5.gene_embedding_max_cells','module5.lens_max_epochs','module5.lens_spots_per_epoch','module5.lens_null_cells','module5.lens_background','module5.lens_absolute_position',
)

STAGE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    'load_inputs': (),
    'preprocess': ('module1.use_input_nucleus_flags','module1.remove_foreign_transcripts','module1.foreign_max_relative_expression'),
    'build_anndata': _BUILD_STAGE_KEYS + (
        'io.drop_embeddings',
        'io.final_keep_columns',
    ),
    'annotate': (
        'annotation.enable','annotation.method','annotation.evidence_axis_source','annotation.evidence_axis_q','annotation.evidence_axis_validation_cells','annotation.evidence_axis_smooth_transcripts','annotation.evidence_axis_radial_bands','annotation.evidence_label_rule','annotation.evidence_validation_folds','annotation.evidence_max_cells','annotation.evidence_min_gene_transcripts','annotation.evidence_marker_margin','annotation.evidence_min_resolution','annotation.evidence_bootstrap','annotation.evidence_seed','annotation.reference_exclude_genes_file','annotation.apex_omm_specificity_filter','annotation.apex_omm_min_log2fc','annotation.apex_omm_min_delta','annotation.rnalocate_species','annotation.rnalocate_augment_er',
    ),
    'write_outputs': (
        'io.final_format','io.h5ad_compression','io.write_csv_copy','io.write_diagnostic_tables',
        'io.write_diagnostic_csv','io.write_diagnostic_parquet',
        'io.write_partitioned_dataset','io.drop_embeddings','io.final_keep_columns',
    ),
}



# --- Module-level param maps for fine-grained resume within build_anndata ---
MODULE_PARAM_MAP: Dict[str, Tuple[str, ...]] = {
    # m1..m6 correspond to pipeline internal modules inside build_anndata
    'm1': (
        'module1.include_directional','module1.use_input_nucleus_flags','module1.winsor_perc_low','module1.winsor_perc_high','module1.boundary_distance','module1.remove_foreign_transcripts','module1.foreign_max_relative_expression',
    ),
    'm2': (
        'module2.top_g','module2.bow_knn_k','module2.max_rows_per_cell_for_svd','module2.max_svd_rows','module2.svd_n_components','module2.bow_weight_mode','module2.use_tfidf','module2.tf_norm','module2.idf_mode','module2.parallel_chunk_cells','module2.svd_sample_cells',
    ),
    'm3': (
        'module3.min_segment_points','module3.n_segmentation_features','module3.spatial_connectivity_k','module3.spatial_radius_quantile','module3.spatial_radius_factor','module3.spatial_smoothing_iterations','module3.spatial_smoothing_min_fraction','module3.spatial_min_component_size','module3.spatial_compactness','module3.target_sites_per_metaspot','module3.min_metaspots_per_cell','module3.max_metaspots_per_cell','module3.use_z_location','module3.segmentation_geometry','module3.segmentation_spatial_share',
    ),
    # m4 builds adata1 from point features; no direct params, but depends on m1..m3
    'm5': (
        'module5.random_seed','module5.metadomain_q','module5.gene_embedding_pair_radius','module5.gene_embedding_neighbour_radius','module5.gene_embedding_scale','module5.gene_embedding_dim','module5.gene_embedding_min_gene_count','module5.gene_embedding_pseudo_count','module5.gene_embedding_max_cells','module5.lens_max_epochs','module5.lens_spots_per_epoch','module5.lens_null_cells','module5.lens_background','module5.lens_absolute_position',
    ),
    'm6': (),
}


def _get_by_path(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def stage_config_subset(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    keys = STAGE_PARAM_MAP.get(stage, ())
    sub: Dict[str, Any] = {}
    for k in keys:
        sub[k] = _get_by_path(cfg, k)
    if stage == 'build_anndata':
        sub['__build_pipeline_schema_version__'] = BUILD_PIPELINE_SCHEMA_VERSION
    elif stage == 'preprocess':
        sub['__preprocess_schema_version__'] = PREPROCESS_SCHEMA_VERSION
    elif stage == 'annotate':
        sub['__annotation_schema_version__'] = ANNOTATION_SCHEMA_VERSION
    return sub


def fingerprint_stage(
    cfg: Dict[str, Any],
    stage: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    sub = stage_config_subset(cfg, stage)
    if context:
        sub["__runtime_context__"] = dict(context)
    try:
        payload = json.dumps(sub, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        payload = str(sub)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def module_config_subset(cfg: Dict[str, Any], module: str) -> Dict[str, Any]:
    keys = MODULE_PARAM_MAP.get(module, ())
    sub: Dict[str, Any] = {}
    for k in keys:
        sub[k] = _get_by_path(cfg, k)
    sub['__module_schema_version__'] = MODULE_SCHEMA_VERSIONS.get(module, 1)
    return sub


def fingerprint_module(
    cfg: Dict[str, Any],
    module: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    sub = module_config_subset(cfg, module)
    if context:
        sub["__runtime_context__"] = dict(context)
    try:
        payload = json.dumps(sub, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        payload = str(sub)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def fingerprint_modules_set(
    cfg: Dict[str, Any],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    mods = ('m1','m2','m3','m5','m6')
    return {
        m: fingerprint_module(cfg, m, context=context)
        for m in mods
    }
