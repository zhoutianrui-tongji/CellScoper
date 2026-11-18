from .runtime import _coerce_pos_int, _joblib_backend, _MAX_SAFE_BLAS_THREADS, _preferred_n_jobs, _replace_frame_columns, _replace_obs_columns
from typing import Any, Callable, cast, Dict, List, Literal, Mapping, Optional, Tuple, Union
import functools
import logging as _logging
import os
import anndata as ad
import numpy as np
import pandas as pd

from .checkpoints import (
    compact_string_columns,
    write_annotated_transcripts,
    io_switches,
    set_io_switches,
    _atomic_dataframe_write,
    _atomic_write_adata,
    _checkpoint_path,
    _intermediate_dir,
    _io_should_write_figures,
    _load_adata_checkpoint,
    _load_dataframe_checkpoint,
    _load_late_point_resume_checkpoint,
    _load_module3_membership_checkpoint,
    _load_state,
    _quarantine_stale_file,
    _resume_annotation_outputs,
    _resume_build_outputs,
    _resume_preprocessed_inputs,
    _resume_raw_inputs,
    _sanitize_obs_for_h5ad,
    _save_adata,
    _save_dataframe,
    _save_fig,
    _save_module3_membership_checkpoint,
    _save_state,
)
from .run_figures import FIGURES_DIR, plot_lens_training
from .composition import _LocalCompositionModel, build_composition_embeddings
from .config import fingerprint_modules_set, fingerprint_stage, load_params_yaml, resolve_annotation_configuration
from .geometry import build_point_geometry_v2
from .io import ensure_dir, load_checkpoint, read_boundaries, read_spatial_transcripts, save_checkpoint
from .keys import COMPACT_STRING_COLS, METASPOT_COL, METASPOT_LOCAL_COL, LENS_REP, TRANSCRIPT_WORKING_COLS
from .logging_utils import setup_logger
from .metadomains import _compute_gene_embedding, build_adata2_from_adata1_and_pointdf, run_module5_simple
from .metaspots import (
    _metaspot_segmentation_audit,
    _validate_adata1_point_df_alignment,
    build_adata1,
    per_cell_clustering,
)
from .outputs import _atomic_partitioned_parquet_write, _compact_final_adata1_for_write, _write_output_guide
from .provenance import (
    atomic_write_json,
    build_input_provenance,
    bundled_reference_provenance,
    sha256_mapping,
    source_tree_provenance,
)
from .reference import _load_reference_scores
from .spatial_gene_embedding import GENE_EMBEDDING_REP


def load_inputs(
    spatial_path: str,
    cell_boundaries_path: str,
    nucleus_boundaries_path: str,
    out_dir: str,
    resume: bool,
    save_intermediate: bool,
    *,
    reuse_cached_inputs: Optional[bool] = None,
):
    """Load raw inputs and optionally maintain raw-input checkpoints.

    ``resume`` controls whether raw checkpoints are persisted. Reuse is a
    separate decision: when provenance selects ``load_inputs`` as the first
    stage to rerun, existing checkpoints are stale and must be overwritten
    from the source paths.
    """
    cp_spatial = _checkpoint_path(out_dir, "spatial")
    cp_cell = _checkpoint_path(out_dir, "cell_boundaries")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_boundaries")
    cache_inputs = bool(resume or save_intermediate)
    reuse_cached_inputs = (
        bool(resume)
        if reuse_cached_inputs is None
        else bool(reuse_cached_inputs)
    )

    if reuse_cached_inputs and os.path.exists(cp_spatial):
        spatial_df = load_checkpoint(cp_spatial)
    else:
        spatial_df = read_spatial_transcripts(spatial_path)
        if cache_inputs:
            save_checkpoint(spatial_df, cp_spatial)

    if reuse_cached_inputs and os.path.exists(cp_cell):
        cell_df = load_checkpoint(cp_cell)
    else:
        cell_df = read_boundaries(cell_boundaries_path)
        if cache_inputs:
            save_checkpoint(cell_df, cp_cell)

    if reuse_cached_inputs and os.path.exists(cp_nuc):
        nucleus_df = load_checkpoint(cp_nuc)
    else:
        nucleus_df = read_boundaries(nucleus_boundaries_path)
        if cache_inputs:
            save_checkpoint(nucleus_df, cp_nuc)

    return spatial_df, cell_df, nucleus_df

def preprocess(
    spatial_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    nucleus_df: pd.DataFrame,
    show_internal_progress: bool = False,
    use_input_nucleus_flags: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align input cells without leaking unused nuclear truth into selection.

    When input nuclear flags are part of Module 1, retain the historical
    requirement that a cell contain both nuclear and non-nuclear transcripts.
    When geometry is recomputed from polygons, select only on membership in the
    three input tables; ``overlaps_nucleus`` then has no influence on the sample.
    """
    del show_internal_progress  # reserved for future streaming validation
    # Per-row work runs on integer cell codes: sets and string groupbys over the
    # ~3e8 rows of a Xenium 5K slide took minutes.
    row_cell, cell_values = pd.factorize(spatial_df["cell_id"])
    common_cells = (
        set(cell_values)
        & set(cell_df["cell_id"].dropna())
        & set(nucleus_df["cell_id"].dropna())
    )

    if use_input_nucleus_flags:
        if "overlaps_nucleus" not in spatial_df.columns:
            raise ValueError(
                "spatial input is missing overlaps_nucleus while "
                "module1.use_input_nucleus_flags=true"
            )
        # Two distinct non-missing flag values per cell, as nunique(dropna=True) == 2.
        flag_codes, _ = pd.factorize(spatial_df["overlaps_nucleus"])
        seen = np.zeros((len(cell_values), max(int(flag_codes.max()) + 1, 1)), dtype=bool)
        present = (row_cell >= 0) & (flag_codes >= 0)
        seen[row_cell[present], flag_codes[present]] = True
        common_cells &= set(cell_values[seen.sum(axis=1) == 2])

    keep_cell = pd.Index(cell_values).isin(list(common_cells))
    spatial_df = spatial_df[(row_cell >= 0) & keep_cell[np.maximum(row_cell, 0)]].copy()
    cell_df = cell_df[cell_df["cell_id"].isin(common_cells)].copy()
    nucleus_df = nucleus_df[nucleus_df["cell_id"].isin(common_cells)].copy()

    return spatial_df, cell_df, nucleus_df


def _save_lens_training(lens_uns, out_dir: str) -> None:
    """写出 LENS 训练曲线：module5_lens_training.csv（每 epoch 一行）和同名 PNG。

    输入为 adata1.uns["lens"]；没有 LENS 或旧版缺 train_nll 时跳过。只写文件，不影响任何结果。
    """
    if not lens_uns or "train_nll_history" not in lens_uns:
        return
    gain = np.asarray(lens_uns["heldout_gain_history"], float)
    table = pd.DataFrame({
        "epoch": np.arange(1, len(gain) + 1),
        "train_nll": np.asarray(lens_uns["train_nll_history"], float),
        "heldout_gain_nats": gain,
        "elapsed_s": np.asarray(lens_uns["seconds_history"], float),
    })
    table["best"] = table["epoch"] == int(lens_uns.get("best_epoch", 0))
    try:
        _atomic_dataframe_write(table, os.path.join(_intermediate_dir(out_dir), "module5_lens_training.csv"),
                                file_format="csv")
        _logging.getLogger("cellscoper").info(
            "LENS training curve saved | epochs=%d | best_epoch=%s | stop=%s",
            len(table), lens_uns.get("best_epoch"), lens_uns.get("stop_reason"))
    except Exception as exc:
        _logging.getLogger("cellscoper").warning("Failed to save LENS training curve: %s", exc)
    try:
        plot_lens_training(out_dir, table, int(lens_uns.get("best_epoch", 0)))
    except Exception as exc:
        _logging.getLogger("cellscoper").warning("Failed to plot LENS training curve: %s", exc)


# Module dependency order: M1 geometry, M2 composition, M3 MetaSpots, M4 adata1, M5 MetaDomains (+ LENS), M6 adata2.
MODULE_ORDER = ["m1", "m2", "m3", "m4", "m5", "m6"]

def build_anndata(spatial_df: pd.DataFrame,
                  cell_df: pd.DataFrame,
                  nucleus_df: pd.DataFrame,
                  show_internal_progress: bool = False,
                  progress_callback: Optional[Callable[[str], None]] = None,
                  out_dir: Optional[str] = None,
                  fingerprint_context: Optional[Mapping[str, Any]] = None,
                  ) -> Tuple[ad.AnnData, Optional[ad.AnnData], pd.DataFrame]:
    # Fine-grained resume within build_anndata
    if out_dir is None:
        out_dir = os.getcwd()
    state_prev = _load_state(out_dir) or {}
    # Fingerprints decide which checkpoints may be reused, so an unreadable
    # configuration must stop the run instead of fingerprinting an empty one.
    cfg = load_params_yaml()
    cur_mod_fps = fingerprint_modules_set(
        cfg,
        context=fingerprint_context,
    )
    # Compact M3 membership is a semantic data artifact. Its reuse depends on
    # the inputs and M1-M3 parameters/schema, not on unrelated source edits in
    # later modules (the normal module fingerprints intentionally include the
    # whole source-tree provenance).
    membership_fingerprint_context = {
        "inputs": (
            fingerprint_context.get("inputs")
            if isinstance(fingerprint_context, Mapping)
            else None
        )
    }
    membership_mod_fps = fingerprint_modules_set(
        cfg,
        context=membership_fingerprint_context,
    )
    prev_mod_fps = state_prev.get("module_fingerprints") or {}
    # First full module run: state has no 'modules_completed' flag yet
    _first_run = not bool(state_prev.get("modules_completed"))
    module_order = MODULE_ORDER
    fingerprint_order = [m for m in module_order if m != "m4"]
    # Determine the earliest changed module in the actual dependency order.
    start_mod = fingerprint_order[0]
    for m in fingerprint_order:
        if prev_mod_fps.get(m) != cur_mod_fps.get(m):
            start_mod = m
            break
    def _mod_lt(m: str, target: str) -> bool:
        return module_order.index(m) < module_order.index(target)
    try:
        import logging as _logging
        _logging.getLogger("cellscoper").info("Fine-grained resume: start module %s", start_mod)
    except Exception:
        pass
    # A downstream checkpoint is reusable only while every earlier module in
    # this invocation also came from its matching checkpoint.  With
    # lite_intermediate enabled, an upstream DataFrame cache may be absent even
    # though a later AnnData cache exists.  Recomputing the upstream module can
    # renumber MetaSpots, so loading any later cache would silently cross-wire
    # expression and geometry.
    _cache_chain_valid = True
    _late_point_df_cache: Optional[pd.DataFrame] = None
    if _mod_lt("m4", start_mod):
        _late_point_df_cache = _load_late_point_resume_checkpoint(
            out_dir,
            spatial_df,
        )
        if _late_point_df_cache is not None:
            _logging.getLogger("cellscoper").info(
                "Resume: using aligned late point-table checkpoint "
                "(module5_final_data) for Modules 1-3"
            )

    # Use unique cell ids to avoid row-level list blow-up (can be hundreds of millions).
    sampled_cell_ids = pd.unique(spatial_df['cell_id'])
    # Read Module 1 geometry options
    _m1_include_dir = True
    # Fallbacks match the packaged params.yaml. Invalid values raise: silently
    # reverting to defaults here could switch nuclear membership back to the
    # input truth column.
    _m1_use_input_nucleus_flags = False
    _m1_winsor = (1, 99)
    _m1_boundary_distance = "vertex"
    _m1 = (cfg or {}).get('module1', {}) or {}
    _m1_boundary_distance = str(
        _m1.get('boundary_distance', _m1_boundary_distance)
    )
    _m1_include_dir = bool(_m1.get('include_directional', _m1_include_dir))
    _m1_use_input_nucleus_flags = bool(
        _m1.get('use_input_nucleus_flags', _m1_use_input_nucleus_flags)
    )
    _m1_winsor = (
        int(_m1.get('winsor_perc_low', _m1_winsor[0])),
        int(_m1.get('winsor_perc_high', _m1_winsor[1])),
    )
    # Module 1: geometry features (resume if unchanged and cache exists)
    point_df = None
    _m1_recomputed = False
    if _mod_lt('m1', start_mod):
        # try load cache
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module1_point_features")
        if pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 1 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscoper").info("Resume: using Module 1 cache (module1_point_features)")
            except Exception:
                pass
            point_df = pd_cache
        elif _late_point_df_cache is not None:
            point_df = _late_point_df_cache
        else:
            if progress_callback: progress_callback("Module 1 · Geometry Features")
            point_df = build_point_geometry_v2(
                transcripts_df=spatial_df,
                nucleus_boundaries_df=nucleus_df,
                cell_boundaries_df=cell_df,
                sampled_cells=sampled_cell_ids,
                include_directional=_m1_include_dir,
                use_input_nucleus_flags=_m1_use_input_nucleus_flags,
                winsor_perc=_m1_winsor,
                show_internal_progress=show_internal_progress,
                boundary_distance=_m1_boundary_distance,
            )
            _save_dataframe(point_df, out_dir, "module1_point_features")
            _m1_recomputed = True
    else:
        if progress_callback: progress_callback("Module 1 · Geometry Features")
        point_df = build_point_geometry_v2(
            transcripts_df=spatial_df,
            nucleus_boundaries_df=nucleus_df,
            cell_boundaries_df=cell_df,
            sampled_cells=sampled_cell_ids,
            include_directional=_m1_include_dir,
            use_input_nucleus_flags=_m1_use_input_nucleus_flags,
            winsor_perc=_m1_winsor,
            show_internal_progress=show_internal_progress,
            boundary_distance=_m1_boundary_distance,
        )
        _save_dataframe(point_df, out_dir, "module1_point_features")
        _m1_recomputed = True
    if _m1_recomputed:
        _cache_chain_valid = False
    # Report Module 1 completion
    try:
        import logging as _logging
        _suf = None if (_first_run and _m1_recomputed) else ("recomputed" if _m1_recomputed else "cache")
        _logging.getLogger("cellscoper").info("Module 1 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m1_recomputed) else ("recomputed" if _m1_recomputed else "cache")
            progress_callback("Module 1 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 2: composition embeddings (+gmm features)
    # Module 2 only selects the columns it needs inside per-cell blocks. Passing
    # the original table avoids copying hundreds of millions of rows up front.
    _m2_source_df = spatial_df

    gmm_cols: List[str] = []
    composition_model: Optional[_LocalCompositionModel] = None
    _m2_recomputed = False
    if _mod_lt('m2', start_mod) and _cache_chain_valid:
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module2_point_df")
        if pd_cache is None and _late_point_df_cache is not None:
            pd_cache = _late_point_df_cache
        if pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 2 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscoper").info("Resume: using Module 2 cache (module2_point_df)")
            except Exception:
                pass
            point_df = pd_cache
            _geom = ['r_signed', 'd_cell_norm', 'd_nuc_norm']
            _ang = ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta']
            _geom = [c for c in _geom + _ang if c in point_df.columns]
            gmm_cols = [c for c in point_df.columns if str(c).startswith("gmm_feat_")]
            if len(gmm_cols) == 0:
                # New low-memory path: Module 2 no longer materializes gmm_feat_*.
                _comp = [c for c in point_df.columns if str(c).startswith('comp_emb')]
                _ld = ['local_density'] if 'local_density' in point_df.columns else []
                gmm_cols = _geom + _comp + _ld
            model_path = os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib")
            try:
                import joblib
                if os.path.exists(model_path):
                    composition_model = joblib.load(model_path)
                    gmm_cols = _geom + composition_model.feature_names + ['local_density']
            except Exception:
                composition_model = None
            if composition_model is None and _late_point_df_cache is None:
                _, gmm_cols, rebuilt_model = build_composition_embeddings(
                    point_df,
                    _m2_source_df,
                    sampled_cell_ids,
                    show_internal_progress=show_internal_progress,
                )
                if isinstance(rebuilt_model, _LocalCompositionModel):
                    composition_model = rebuilt_model
                    try:
                        import joblib
                        joblib.dump(composition_model, model_path)
                    except Exception:
                        pass
        else:
            if progress_callback: progress_callback("Module 2 · Composition Embeddings")
            point_df, gmm_cols, composition_model = build_composition_embeddings(point_df, _m2_source_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
            try:
                import joblib
                joblib.dump(composition_model, os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib"))
            except Exception:
                pass
            _save_dataframe(point_df, out_dir, "module2_point_df")
            _m2_recomputed = True
    else:
        if progress_callback: progress_callback("Module 2 · Composition Embeddings")
        point_df, gmm_cols, composition_model = build_composition_embeddings(point_df, _m2_source_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
        try:
            import joblib
            joblib.dump(composition_model, os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib"))
        except Exception:
            pass
        _save_dataframe(point_df, out_dir, "module2_point_df")
        _m2_recomputed = True
    if _m2_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m2_recomputed) else ("recomputed" if _m2_recomputed else "cache")
        _logging.getLogger("cellscoper").info("Module 2 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m2_recomputed) else ("recomputed" if _m2_recomputed else "cache")
            progress_callback("Module 2 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 3: per-cell MetaSpot segmentation
    # Module 3 options (defaults match the packaged params.yaml)
    _m3 = (load_params_yaml() or {}).get('module3', {}) or {}
    _m3_target_sites_per_metaspot = int(_m3.get('target_sites_per_metaspot', 15))
    _m3_min_metaspots_per_cell = int(_m3.get('min_metaspots_per_cell', 1))
    _m3_max_metaspots_per_cell = (
        int(_m3['max_metaspots_per_cell']) if isinstance(_m3.get('max_metaspots_per_cell'), (int, float)) else None
    )
    _m3_options = dict(
        min_segment_points=int(_m3.get('min_segment_points', 30)),
        n_segmentation_features=int(_m3.get('n_segmentation_features', 24)),
        small_backend_cells=int(_m3.get('small_backend_cells', 12000)),
        sequential_cells=int(_m3.get('sequential_cells', 256)),
        spatial_connectivity_k=int(_m3.get('spatial_connectivity_k', 8)),
        spatial_radius_quantile=float(_m3.get('spatial_radius_quantile', 0.9)),
        spatial_radius_factor=float(_m3.get('spatial_radius_factor', 1.5)),
        spatial_smoothing_iterations=int(_m3.get('spatial_smoothing_iterations', 1)),
        spatial_smoothing_min_fraction=float(_m3.get('spatial_smoothing_min_fraction', 0.6)),
        spatial_min_component_size=(
            int(_m3['spatial_min_component_size']) if isinstance(_m3.get('spatial_min_component_size'), (int, float)) else None
        ),
        spatial_compactness=float(_m3.get('spatial_compactness', 1.0)),
        target_sites_per_metaspot=_m3_target_sites_per_metaspot,
        min_metaspots_per_cell=_m3_min_metaspots_per_cell,
        max_metaspots_per_cell=_m3_max_metaspots_per_cell,
        parallel_chunk_cells=int(_m3['parallel_chunk_cells']) if isinstance(_m3.get('parallel_chunk_cells'), (int, float)) else None,
        parallel_n_jobs=int(_m3['parallel_n_jobs']) if isinstance(_m3.get('parallel_n_jobs'), (int, float)) else None,
        use_z_location=bool(_m3.get('use_z_location', True)),
        segmentation_geometry=str(_m3.get('segmentation_geometry', 'legacy')),
        segmentation_spatial_share=float(_m3.get('segmentation_spatial_share', 1.0 / 3.0)),
    )
    import time as _time
    _t_m3 = _time.perf_counter()
    _m3_recomputed = False
    _m3_compact_reused = False
    pd_cache = None
    if _mod_lt('m3', start_mod) and _cache_chain_valid:
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module3_point_df")
        if pd_cache is None and _late_point_df_cache is not None:
            pd_cache = _late_point_df_cache
    if pd_cache is None:
        pd_cache = _load_module3_membership_checkpoint(
            point_df,
            out_dir,
            module_fingerprints=membership_mod_fps,
        )
        _m3_compact_reused = pd_cache is not None
    if pd_cache is not None:
        if progress_callback:
            progress_callback("Resume: load Module 3 membership")
        point_df = pd_cache
        _logging.getLogger("cellscoper").info(
            "Resume: using %s Module 3 checkpoint",
            "compact membership" if _m3_compact_reused else "full point-table",
        )
    else:
        if progress_callback: progress_callback("Module 3 · MetaSpot Segmentation")
        point_df = per_cell_clustering(
            point_df,
            gmm_cols,
            composition_model=composition_model,
            show_internal_progress=show_internal_progress,
            **_m3_options,
        )
        _save_dataframe(point_df, out_dir, "module3_point_df")
        _m3_recomputed = True
    if _m3_recomputed:
        _cache_chain_valid = False
    try:
        _cells = int(point_df['cell_id'].nunique()) if 'cell_id' in point_df.columns else None
    except Exception:
        _cells = None
    _dt_m3 = _time.perf_counter() - _t_m3
    if show_internal_progress:
        msg = f"<< Done: Module 3 (MetaSpot segmentation) in {_dt_m3:0.2f}s" + (f" (cells={_cells})" if _cells is not None else "")
        print(msg)
    try:
        import logging as _logging
        _logging.getLogger("cellscoper").info(
            "Module 3 segmentation done in %.2fs%s",
            _dt_m3,
            (f" (cells={_cells})" if _cells is not None else ""),
        )
    except Exception:
        pass
    _audit_t0 = _time.perf_counter()
    if progress_callback:
        try:
            progress_callback("Module 3 · Audit segmentation quality")
        except Exception:
            pass
    _logging.getLogger("cellscoper").info(
        "Module 3 segmentation audit started | rows=%d | cells=%s",
        len(point_df),
        _cells,
    )
    try:
        segmentation_audit = (
            _load_dataframe_checkpoint(
                _intermediate_dir(out_dir),
                "module3_metaspot_segmentation_audit",
            )
            if _m3_compact_reused
            else None
        )
        if segmentation_audit is None:
            segmentation_audit = _metaspot_segmentation_audit(
                point_df,
                target_sites_per_metaspot=_m3_target_sites_per_metaspot,
                min_metaspots_per_cell=_m3_min_metaspots_per_cell,
                max_metaspots_per_cell=_m3_max_metaspots_per_cell,
            )
            _save_dataframe(
                segmentation_audit,
                out_dir,
                "module3_metaspot_segmentation_audit",
                diagnostic=True,
            )
        _logging.getLogger("cellscoper").info(
            "Module 3 audit: MetaSpots/cell median=%.1f, "
            "actual/requested median=%.3f, "
            "nucleus-mixed transcript fraction=%.6f",
            float(segmentation_audit["actual_metaspots"].median()),
            float(
                segmentation_audit[
                    "actual_to_requested_ratio"
                ].median()
            ),
            float(
                segmentation_audit["mixed_transcripts"].sum()
                / max(
                    1,
                    segmentation_audit["n_transcripts"].sum(),
                )
            ),
        )
        _logging.getLogger("cellscoper").info(
            "Module 3 segmentation audit done in %.2fs",
            _time.perf_counter() - _audit_t0,
        )
    except Exception as exc:
        _logging.getLogger("cellscoper").warning(
            "Module 3 segmentation audit failed after %.2fs: %s",
            _time.perf_counter() - _audit_t0,
            exc,
        )
    # This compact membership contract is mandatory even when bulky
    # intermediate tables are disabled.  It is sufficient to resume or audit
    # segmentation without serializing every Module 1/2 feature column.
    if _m3_recomputed:
        try:
            _save_module3_membership_checkpoint(
                point_df,
                out_dir,
                module_fingerprints=membership_mod_fps,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not persist the mandatory Module 3 MetaSpot "
                "membership checkpoint"
            ) from exc
    # Save histogram of metaspots per cell
    if not _m3_compact_reused:
        try:
            import matplotlib.pyplot as plt
            vc = point_df.groupby('cell_id')[METASPOT_LOCAL_COL].nunique()
            fig = plt.figure(figsize=(6,4))
            ax = vc.plot.hist(bins=50)
            ax.set_xlabel("n_metaspots per cell"); ax.set_ylabel("count"); plt.tight_layout()
            _save_fig(fig, out_dir or os.getcwd(), "module3_metaspots_per_cell_hist")
            plt.close(fig)
        except Exception:
            pass
    if progress_callback:
        try:
            progress_callback("Module 3 · Done")
        except Exception:
            pass
    # 输入表只有 Module 1、2 用到；调用方还持有引用，就地清空才能真正释放
    spatial_df.drop(columns=list(spatial_df.columns), inplace=True)
    _m2_source_df = None
    point_df = compact_string_columns(point_df)
    from .scalable_execution import resolve_execution_plan

    execution_plan: Dict[str, Any] = resolve_execution_plan(
        cfg,
        n_metaspots=int(point_df[METASPOT_COL].nunique()),
        n_transcripts=int(len(point_df)),
    ).to_dict()
    if not bool(execution_plan.get("full_diagnostics", True)):
        _logging.getLogger("cellscoper").info(
            "Streaming profile: disabling full diagnostic tables; "
            "downstream figures follow io.write_intermediate_figures=%s",
            _io_should_write_figures(),
        )
    # Module 4: adata1 from point_df (derived; rerun if m1–m3 changed)
    _m4_recomputed = False
    if (
        (_mod_lt('m4', start_mod) and _cache_chain_valid)
        or _m3_compact_reused
    ):
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module4_adata1.h5ad"))
        if a1_cache is not None:
            aligned, alignment_reason = _validate_adata1_point_df_alignment(
                a1_cache,
                point_df,
            )
            if not aligned:
                _logging.getLogger("cellscoper").warning(
                    "Rejecting Module 4 cache: %s",
                    alignment_reason,
                )
                a1_cache = None
        if a1_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 4 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscoper").info("Resume: using Module 4 cache (module4_adata1.h5ad)")
            except Exception:
                pass
            adata1 = a1_cache
        else:
            if progress_callback: progress_callback("Module 4 · Build Per-cell Feature Table (adata1)")
            adata1 = build_adata1(point_df, show_internal_progress=show_internal_progress)
            _save_adata(adata1, out_dir, "module4_adata1")
            _m4_recomputed = True
    else:
        if progress_callback: progress_callback("Module 4 · Build Per-cell Feature Table (adata1)")
        adata1 = build_adata1(point_df, show_internal_progress=show_internal_progress)
        _save_adata(adata1, out_dir, "module4_adata1")
        _m4_recomputed = True
    aligned, alignment_reason = _validate_adata1_point_df_alignment(
        adata1,
        point_df,
    )
    if not aligned:
        raise RuntimeError(
            "Module 4 MetaSpot alignment check failed: "
            f"{alignment_reason}. Refusing to continue with mismatched "
            "expression and geometry."
        )
    _logging.getLogger("cellscoper").info(
        "Module 4 MetaSpot alignment validated (%d MetaSpots)",
        adata1.n_obs,
    )
    if _m4_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m4_recomputed) else ("recomputed" if _m4_recomputed else "cache")
        _logging.getLogger("cellscoper").info("Module 4 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m4_recomputed) else ("recomputed" if _m4_recomputed else "cache")
            progress_callback("Module 4 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    if int(execution_plan["n_metaspots"]) != int(adata1.n_obs):
        raise RuntimeError(
            "Execution-plan MetaSpot count differs from Module 4 AnnData"
        )
    adata1.uns["execution_plan"] = dict(execution_plan)
    try:
        atomic_write_json(
            os.path.join(
                _intermediate_dir(out_dir or os.getcwd()),
                "execution_plan.json",
            ),
            execution_plan,
        )
    except Exception as exc:
        _logging.getLogger("cellscoper").warning(
            "Execution-plan manifest could not be written: %s",
            exc,
        )
    _logging.getLogger("cellscoper").info(
        "Scale-aware execution plan | profile=%s | MetaSpots=%d",
        execution_plan["effective_profile"],
        int(adata1.n_obs),
    )


    # Module 5: GE, LENS and MetaDomains (allow resume)
    _m5_recomputed = False
    if _mod_lt('m5', start_mod) and _cache_chain_valid:
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module5_adata1.h5ad"))
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module5_point_df")
        if a1_cache is not None and pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 5 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscoper").info("Resume: using Module 5 cache (module5_adata1.h5ad, module5_point_df)")
            except Exception:
                pass
            adata1 = a1_cache
            point_df = pd_cache
            diag_simple = None
        else:
            if progress_callback: progress_callback("Module 5 · LENS and MetaDomains")
            adata1, diag_simple = run_module5_simple(
                adata1,
                point_df,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
            )
            _m5_recomputed = True
    else:
        if progress_callback: progress_callback("Module 5 · LENS and MetaDomains")
        adata1, diag_simple = run_module5_simple(
            adata1,
            point_df,
            show_internal_progress=show_internal_progress,
            progress_callback=progress_callback,
        )
        _m5_recomputed = True
    if _m5_recomputed:
        _cache_chain_valid = False
        _save_lens_training(adata1.uns.get("lens"), out_dir or os.getcwd())
    # Stats: transcripts count per meta-domain
    try:
        vc = point_df.groupby("meta-domain")["transcript_id"].count()
        stats = vc.describe().to_frame(name="value")
        _save_dataframe(stats.reset_index(), out_dir or os.getcwd(), "module5_domain_transcript_stats")
    except Exception:
        pass
    # Expose the canonical MetaSpot identifier for downstream joins.
    if METASPOT_COL not in adata1.obs.columns:
        adata1.obs = adata1.obs.copy()
        adata1.obs[METASPOT_COL] = adata1.obs.index.astype(str)
    # Attach meta-domain to point_df
    for column in ("meta-domain", "meta-domain_score"):
        if column in point_df.columns:
            point_df = point_df.drop(columns=[column])  # ensure fresh
    point_df = point_df.merge(
        adata1.obs[[METASPOT_COL, "meta-domain"]],
        on=METASPOT_COL,
        how="left",
    )
    try:
        _save_dataframe(point_df, out_dir, "module5_point_df")
    except Exception:
        pass
    try:
        import logging as _logging
        _suf = None if (_first_run and _m5_recomputed) else ("recomputed" if _m5_recomputed else "cache")
        _logging.getLogger("cellscoper").info("Module 5 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m5_recomputed) else ("recomputed" if _m5_recomputed else "cache")
            progress_callback("Module 5 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    # Module 6: Build adata2 from adata1 + meta-domains
    # Module 6: adata2 (allow resume)
    _m6_recomputed = False
    if _mod_lt('m6', start_mod) and _cache_chain_valid:
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module6_adata2.h5ad"))
        if a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 6 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscoper").info("Resume: using Module 6 cache (module6_adata2.h5ad)")
            except Exception:
                pass
            adata2 = a2_cache
        else:
            if progress_callback: progress_callback("Module 6 · Aggregate Domains into Cluster Feature Table (adata2)")
            adata2 = build_adata2_from_adata1_and_pointdf(
                adata1,
                point_df,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
            )
            _m6_recomputed = True
    else:
        if progress_callback: progress_callback("Module 6 · Aggregate Domains into Cluster Feature Table (adata2)")
        adata2 = build_adata2_from_adata1_and_pointdf(
            adata1,
            point_df,
            show_internal_progress=show_internal_progress,
            progress_callback=progress_callback,
        )
        _m6_recomputed = True
    if _m6_recomputed:
        _cache_chain_valid = False
    try:
        _cfg = load_params_yaml()
        save_shape = bool((_cfg.get('module6', {}) if _cfg else {}).get('save_shape_summary', True))
        if save_shape:
            # log basic adata2 shape
            df = pd.DataFrame({"n_obs":[adata2.n_obs],"n_vars":[adata2.n_vars]})
            _save_dataframe(df, out_dir or os.getcwd(), "module6_adata2_shape")
    except Exception:
        pass
    try:
        import logging as _logging
        _suf = None if (_first_run and _m6_recomputed) else ("recomputed" if _m6_recomputed else "cache")
        _logging.getLogger("cellscoper").info("Module 6 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m6_recomputed) else ("recomputed" if _m6_recomputed else "cache")
            progress_callback("Module 6 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    final_csv = point_df.copy()
    return adata1, adata2, final_csv


def _run_pipeline_impl(
    spatial_path: str,
    cell_boundaries_path: str,
    nucleus_boundaries_path: str,
    out_dir: str,
    gold_data: Optional[Union[str, pd.DataFrame]] = None,
    resume: bool = True,
    save_intermediate: Optional[bool] = None,
    lite_intermediate: Optional[bool] = None,
    final_format: Optional[Literal["parquet","csv"]] = None,
    compression: Optional[str] = None,
    resume_from: Optional[Literal[
        "load_inputs",
        "preprocess",
        "build_anndata",
        "annotate",
        "write_outputs",
    ]] = None,
    show_internal_progress: bool = False,
    enable_annotation: Optional[bool] = None,
    annotation_method: Optional[str] = None,
    resume_policy: Optional[Literal["auto","minimal","force"]] = None,
    dry_run_diff: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    input_provenance: Optional[Mapping[str, Any]] = None,
    ) -> dict:
    """
    Execute the CellScoper pipeline.

    Returns dict with output file paths.
    """
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)
    logger.info("CellScoper pipeline start | out_dir=%s", out_dir)
    # Load config early to allow overriding some runtime I/O defaults
    cfg = load_params_yaml()
    write_csv_copy = True
    io_cfg: Dict[str, Any] = {}
    try:
        io_cfg = cfg.get('io', {}) if cfg else {}
        # Explicit Python/CLI arguments win. YAML supplies values only when the
        # caller left an argument unset.
        if final_format is None and isinstance(io_cfg.get('final_format'), str):
            final_format = str(io_cfg.get('final_format')).lower()
        if compression is None and isinstance(io_cfg.get('h5ad_compression'), str):
            compression = str(io_cfg.get('h5ad_compression'))
        if 'write_csv_copy' in io_cfg:
            write_csv_copy = bool(io_cfg.get('write_csv_copy'))
        if lite_intermediate is None and 'lite_intermediate' in io_cfg:
            lite_intermediate = bool(io_cfg.get('lite_intermediate'))
        if save_intermediate is None and 'write_intermediate' in io_cfg:
            save_intermediate = bool(io_cfg.get('write_intermediate'))
        if resume_from is None and isinstance(
            io_cfg.get('resume_from'),
            (str, type(None)),
        ):
            rf = (io_cfg.get('resume_from') or '').strip()
            resume_from = rf if rf else resume_from
        if resume_policy is None and isinstance(io_cfg.get('resume_policy'), str):
            resume_policy = str(io_cfg.get('resume_policy')).lower()
    except Exception as exc:
        logger.warning(
            "Could not resolve runtime I/O configuration: %s",
            exc,
            exc_info=True,
        )
    final_format = str(final_format or "csv").lower()
    if final_format not in {"csv", "parquet"}:
        raise ValueError("final_format must be 'csv' or 'parquet'")
    compression = str(compression or "lzf")
    lite_intermediate = bool(
        False if lite_intermediate is None else lite_intermediate
    )
    save_intermediate = bool(
        True if save_intermediate is None else save_intermediate
    )
    resume_policy = str(resume_policy or "minimal").lower()
    if resume_policy not in {"auto", "minimal", "force"}:
        raise ValueError(
            "resume_policy must be one of: auto, minimal, force"
        )
    try:
        runtime_cfg = cfg.get('runtime', {}) if cfg else {}
    except Exception:
        runtime_cfg = {}
    # Annotation toggles/method from YAML (default on, method=evidence)
    try:
        ann_cfg = cfg.get('annotation', {}) if cfg else {}
        if enable_annotation is None and 'enable' in ann_cfg:
            enable_annotation = bool(ann_cfg.get('enable'))
        if annotation_method is None and 'method' in ann_cfg:
            _m = str(ann_cfg.get('method') or "").strip().lower()
            if _m:
                annotation_method = _m
    except Exception as exc:
        logger.warning(
            "Could not resolve annotation runtime settings: %s",
            exc,
        )
    enable_annotation = bool(
        True if enable_annotation is None else enable_annotation
    )
    annotation_method = str(annotation_method or "evidence").lower()
    logger.info(
        "Effective runtime | config=%s | config_sha256=%s | resume=%s | "
        "resume_policy=%s | save_intermediate=%s | lite_intermediate=%s | "
        "final_format=%s | compression=%s | annotation=%s:%s",
        os.environ.get("CELLSCOPER_CONFIG", "(package default)"),
        sha256_mapping(cfg),
        bool(resume),
        resume_policy,
        save_intermediate,
        lite_intermediate,
        final_format,
        compression,
        enable_annotation,
        annotation_method,
    )
    # Set global I/O switches for helpers
    # I/O switches for the checkpoint helpers; figures default to on unless disabled
    write_figures = bool(io_cfg.get('write_intermediate_figures', True)) if io_cfg else True
    set_io_switches(bool(save_intermediate), bool(lite_intermediate), write_figures)
    # Cap BLAS/OMP threads; allow runtime.blas_threads in params.yaml to take effect.
    # OpenBLAS cannot safely exceed the package's compiled thread-table limit.
    try:
        _requested_blas_threads = (
            _coerce_pos_int(runtime_cfg.get('blas_threads'))
            if runtime_cfg
            else None
        )
        _threaded_outer_parallelism = (
            _joblib_backend().strip().lower() == "threading"
            and _preferred_n_jobs() > 1
        )
        _rt_blas_threads = (
            1
            if _threaded_outer_parallelism
            else (
                min(_requested_blas_threads, _MAX_SAFE_BLAS_THREADS)
                if _requested_blas_threads
                else None
            )
        )
        if _threaded_outer_parallelism and (
            _requested_blas_threads is None or _requested_blas_threads > 1
        ):
            logger.warning(
                "Threaded outer parallelism requires single-threaded BLAS; "
                "using 1 instead of runtime.blas_threads=%s.",
                _requested_blas_threads,
            )
        elif _requested_blas_threads and _requested_blas_threads > _MAX_SAFE_BLAS_THREADS:
            logger.warning(
                "runtime.blas_threads=%d exceeds the OpenBLAS safe limit %d; "
                "using %d.",
                _requested_blas_threads,
                _MAX_SAFE_BLAS_THREADS,
                _rt_blas_threads,
            )
        if _rt_blas_threads:
            for _env_k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[_env_k] = str(_rt_blas_threads)
            if runtime_cfg.get('use_threadpoolctl'):
                try:
                    from threadpoolctl import threadpool_limits
                    threadpool_limits(int(_rt_blas_threads))
                except Exception:
                    pass
        else:
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
            os.environ.setdefault("MKL_NUM_THREADS",     "1")
            os.environ.setdefault("OMP_NUM_THREADS",     "1")
            os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
        # Silence JAX GPU reminder by forcing CPU backend
        os.environ.setdefault("JAX_PLATFORM_NAME",    "cpu")
        os.environ.setdefault("JAX_PLATFORMS",        "cpu")
    except Exception:
        pass
    logger.info(
        "Numerical threading | backend=%s | outer_jobs=%d | blas=%s | "
        "numba=%s(requested=%s) | scanpy=%s(requested=%s)",
        _joblib_backend(),
        _preferred_n_jobs(),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMBA_NUM_THREADS"),
        runtime_cfg.get("numba_threads"),
        os.environ.get("SCANPY_N_JOBS"),
        runtime_cfg.get("scanpy_n_jobs"),
    )
    # Suppress only stable third-party compatibility notices. Runtime warnings
    # and numerical problems must remain observable.
    try:
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"anndata.*")
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn.utils.deprecation")
    except Exception:
        pass
    # Downgrade Lightning/PL logs to ERROR to avoid hardware banners
    try:
        import logging as _logging
        for _name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
            try:
                _logging.getLogger(_name).setLevel(_logging.ERROR)
            except Exception:
                pass
    except Exception:
        pass

    # Freeze current inputs, references, and source identity into resume
    # fingerprints so an in-place file or code change cannot reuse stale
    # checkpoints merely because params.yaml is unchanged.
    hash_inputs = bool(io_cfg.get("provenance_hash_inputs", True))
    if input_provenance is None:
        input_provenance = build_input_provenance(
            spatial_path=spatial_path,
            cell_boundaries_path=cell_boundaries_path,
            nucleus_boundaries_path=nucleus_boundaries_path,
            gold_data=gold_data,
            hash_inputs=hash_inputs,
        )
    else:
        input_provenance = dict(input_provenance)
    package_dir = os.path.dirname(__file__)
    reference_provenance = bundled_reference_provenance(
        package_dir,
        hash_content=hash_inputs,
    )
    code_provenance = source_tree_provenance(package_dir)
    raw_input_provenance = {
        key: input_provenance.get(key)
        for key in (
            "spatial",
            "cell_boundaries",
            "nucleus_boundaries",
        )
    }
    build_fingerprint_context = {
        "inputs": raw_input_provenance,
        "code": code_provenance,
    }
    annotation_fingerprint_context = {
        "gold_data": input_provenance.get("gold_data"),
        "references": reference_provenance,
        "code": code_provenance,
    }
    stage_fingerprint_contexts = {
        "load_inputs": build_fingerprint_context,
        "preprocess": build_fingerprint_context,
        "build_anndata": build_fingerprint_context,
        "annotate": annotation_fingerprint_context,
        "write_outputs": {"code": code_provenance},
    }
    for input_name, record in input_provenance.items():
        if not isinstance(record, Mapping):
            logger.info("Input provenance | name=%s | value=%s", input_name, record)
            continue
        digest = str(record.get("sha256", "not-computed"))
        logger.info(
            "Input provenance | name=%s | path=%s | size_bytes=%s | "
            "sha256=%s",
            input_name,
            record.get("path", record.get("kind", "in-memory")),
            record.get("size_bytes", record.get("n_rows")),
            digest,
        )
    logger.info(
        "Source provenance | python_source_sha256=%s | python_files=%s",
        code_provenance.get("python_source_sha256"),
        code_provenance.get("n_python_files"),
    )
    for reference_name, record in reference_provenance.items():
        if isinstance(record, Mapping):
            logger.info(
                "Reference provenance | name=%s | size_bytes=%s | "
                "sha256=%s",
                reference_name,
                record.get("size_bytes"),
                record.get("sha256", "not-computed"),
            )

    # Determine minimal rerun start using config and provenance fingerprints.
    stage_order = ["load_inputs", "preprocess", "build_anndata", "annotate", "write_outputs"]
    start_from = None
    cur_fps = {
        stage: fingerprint_stage(
            cfg,
            stage,
            context=stage_fingerprint_contexts.get(stage),
        )
        for stage in stage_order
    }
    try:
        cur_mod_fps = fingerprint_modules_set(
            cfg,
            context=build_fingerprint_context,
        )
    except Exception as exc:
        logger.warning(
            "Could not build module fingerprints: %s",
            exc,
            exc_info=True,
        )
        cur_mod_fps = {}
    st = _load_state(out_dir) or {}
    prev_fps = st.get("fingerprints") or {}
    # default: if explicit resume_from provided, honor it
    if resume_from:
        start_from = resume_from
        logger.info("Resume requested from %s", start_from)
    else:
        # Choose behavior based on resume_policy
        policy = str(resume_policy or "minimal").lower()
        if policy == "force":
            start_from = "load_inputs"
            logger.info("Resume policy=force; starting from %s", start_from)
        elif policy == "auto":
            # continue from last completed + 1 if available
            if st.get("last_completed") in stage_order:
                last = st["last_completed"]
                try:
                    idx = stage_order.index(last)
                    start_from = stage_order[min(idx + 1, len(stage_order) - 1)]
                except Exception:
                    start_from = None
            if start_from:
                logger.info("Resume policy=auto; continuing from %s (last completed: %s)", start_from, st.get("last_completed"))
            else:
                logger.info("Resume policy=auto; starting from %s", stage_order[0])
        else:  # minimal (diff-based)
            start_from = None
            for s in stage_order:
                if prev_fps.get(s) != cur_fps.get(s):
                    start_from = s
                    break
            if start_from:
                logger.info(
                    "Resume policy=minimal; configuration/provenance "
                    "fingerprint changed | start_from=%s | previous=%s | "
                    "current=%s",
                    start_from,
                    str(prev_fps.get(start_from, "missing"))[:12],
                    str(cur_fps.get(start_from, "missing"))[:12],
                )
            else:
                # fall back to last_completed if exists
                if st.get("last_completed") in stage_order:
                    last = st["last_completed"]
                    try:
                        idx = stage_order.index(last)
                        start_from = stage_order[min(idx + 1, len(stage_order) - 1)]
                    except Exception:
                        start_from = None
                if start_from:
                    logger.info("Resume policy=minimal; auto-resume from %s (last completed: %s)", start_from, st.get("last_completed"))

    # Dry-run: show diff and planned start, then return immediately
    if dry_run_diff:
        try:
            changes = [s for s in stage_order if prev_fps.get(s) != cur_fps.get(s)]
        except Exception:
            changes = []
        logger.info("Dry-run diff: changed stages=%s; start_from=%s", ",".join(changes) if changes else "(none)", start_from or stage_order[0])
        for stage in stage_order:
            logger.info(
                "Dry-run fingerprint | stage=%s | changed=%s | "
                "previous=%s | current=%s",
                stage,
                prev_fps.get(stage) != cur_fps.get(stage),
                str(prev_fps.get(stage, "missing"))[:12],
                str(cur_fps.get(stage, "missing"))[:12],
            )
        return {
            "dry_run": True,
            "changed_stages": changes,
            "start_from": start_from or stage_order[0],
            "state_file": os.path.join(out_dir, "pipeline_state.json"),
        }

    # Resumed runs reuse the same output directory. Remove only exact files
    # retired by the current output contract so stale visuals cannot be
    # mistaken for results from this invocation.

    stage_to_idx = {name: idx for idx, name in enumerate(stage_order)}
    resume_enabled = bool(resume)
    if resume_enabled:
        resume_stage_key = start_from if isinstance(start_from, str) else None
        if resume_stage_key and resume_stage_key not in stage_to_idx:
            logger.warning("Unknown resume stage '%s'; default to %s", resume_stage_key, stage_order[0])
            resume_stage_key = None
        start_idx = stage_to_idx.get(resume_stage_key, 0) if resume_stage_key else 0
        logger.info("Effective resume start stage: %s", stage_order[start_idx])
        # Nudge CLI progress early so users don't see a long 'Starting pipeline' while loading caches
        try:
            if progress_callback:
                progress_callback(f"Resume from {stage_order[start_idx]}")
        except Exception:
            pass
    else:
        start_idx = 0

    def _should_run(stage_name: str, has_cache: bool) -> bool:
        if not resume_enabled:
            return True
        idx = stage_to_idx[stage_name]
        if idx >= start_idx:
            return True
        return not has_cache

    # ==== Stage 1: Load inputs ====
    spatial_df = cell_df = nucleus_df = None
    adata1 = adata2 = None
    final_data = None

    # If resuming past load_inputs, we still need to load cached inputs; update progress text first.
    if resume_enabled and stage_to_idx["load_inputs"] < start_idx and progress_callback:
        try:
            progress_callback("Loading cached inputs")
        except Exception:
            pass
    raw_inputs_cache = _resume_raw_inputs(out_dir) if (resume_enabled and stage_to_idx["load_inputs"] < start_idx) else None
    if _should_run("load_inputs", raw_inputs_cache is not None):
        if resume_enabled and stage_to_idx["load_inputs"] < start_idx and raw_inputs_cache is None:
            logger.info("Resume requested skip for load_inputs but checkpoint missing → re-running")
        if progress_callback: progress_callback("Loading inputs")
        try:
            spatial_df, cell_df, nucleus_df = load_inputs(
                spatial_path,
                cell_boundaries_path,
                nucleus_boundaries_path,
                out_dir,
                resume,
                save_intermediate,
                # Reaching this branch means the stage must run. Existing raw
                # checkpoints may have a different input fingerprint.
                reuse_cached_inputs=False,
            )
            raw_inputs_cache = (spatial_df, cell_df, nucleus_df)
            _save_state(out_dir, "load_inputs", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage load_inputs done:spatial=%d rows, cells_boundaries=%d, nucleus_boundaries=%d", len(spatial_df), len(cell_df), len(nucleus_df))
        except Exception:
            logger.exception("Stage load_inputs failed")
            raise
    else:
        if raw_inputs_cache is None:
            raise RuntimeError("Missing 'load_inputs' checkpoint; cannot skip this stage.")
        spatial_df, cell_df, nucleus_df = raw_inputs_cache
        logger.info("Resume: skipping load_inputs (checkpoint available)")

    # ==== Stage 2: Preprocess ====
    # If resuming past preprocess, load cached preprocessed data; update progress text first.
    if resume_enabled and stage_to_idx["preprocess"] < start_idx and progress_callback:
        try:
            progress_callback("Loading cached preprocessed inputs")
        except Exception:
            pass
    preproc_cache = _resume_preprocessed_inputs(out_dir) if (resume_enabled and stage_to_idx["preprocess"] < start_idx) else None
    if _should_run("preprocess", preproc_cache is not None):
        if resume_enabled and stage_to_idx["preprocess"] < start_idx and preproc_cache is None:
            logger.info("Resume requested skip for preprocess but checkpoint missing → re-running")
        if progress_callback: progress_callback("Preprocess inputs")
        try:
            module1_cfg = cfg.get("module1", {}) if cfg else {}
            spatial_df, cell_df, nucleus_df = preprocess(
                spatial_df,
                cell_df,
                nucleus_df,
                show_internal_progress=show_internal_progress,
                use_input_nucleus_flags=bool(
                    module1_cfg.get("use_input_nucleus_flags", False)
                ),
            )
            if bool(module1_cfg.get("remove_foreign_transcripts", True)):
                from .run_figures import plot_foreign_removal
                from .spillover import relative_type_expression

                foreign_threshold = float(module1_cfg.get("foreign_max_relative_expression", 0.2))
                relative, cell_codes, cell_type = relative_type_expression(spatial_df, return_cell_types=True)
                keep = relative > foreign_threshold
                if _io_should_write_figures():
                    try:
                        plot_foreign_removal(out_dir, cell_codes, cell_type, ~keep, foreign_threshold)
                    except Exception as exc:  # 诊断图失败不影响运行结果
                        logger.warning("Diagnostic figure 01_foreign_transcripts failed: %s", exc)
                del relative, cell_codes
                logger.info(
                    "Removed %d of %d transcripts (%.1f%%) whose gene is foreign to the cell's type",
                    int((~keep).sum()), len(keep), 100.0 * float((~keep).mean()),
                )
                spatial_df = spatial_df[keep].copy()
            if save_intermediate and not lite_intermediate:
                _save_dataframe(spatial_df, out_dir, "preprocess_spatial")
                _save_dataframe(cell_df, out_dir, "preprocess_cell_boundaries")
                _save_dataframe(nucleus_df, out_dir, "preprocess_nucleus_boundaries")
            try:
                save_checkpoint(spatial_df, _checkpoint_path(out_dir, "spatial_preproc"))
                save_checkpoint(cell_df, _checkpoint_path(out_dir, "cell_preproc"))
                save_checkpoint(nucleus_df, _checkpoint_path(out_dir, "nucleus_preproc"))
            except Exception:
                pass
            _save_state(out_dir, "preprocess", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage preprocess done: spatial=%d rows, cells_boundaries=%d, nucleus_boundaries=%d", len(spatial_df), len(cell_df), len(nucleus_df))
        except Exception:
            logger.exception("Stage preprocess failed")
            raise
    else:
        if preproc_cache is None:
            raise RuntimeError("Missing 'preprocess' checkpoint; cannot skip this stage.")
        spatial_df, cell_df, nucleus_df = preproc_cache
        logger.info("Resume: skipping preprocess (checkpoint available)")

    # ==== Stage 3: Build AnnData + final table ====
    raw_inputs_cache = None  # 原始输入表预处理后不再需要
    build_cache = _resume_build_outputs(out_dir) if (resume_enabled and stage_to_idx["build_anndata"] < start_idx) else None
    build_cache_ready = bool(build_cache and build_cache[0] is not None and build_cache[2] is not None)
    if _should_run("build_anndata", build_cache_ready):
        if resume_enabled and stage_to_idx["build_anndata"] < start_idx and not build_cache_ready:
            logger.info("Resume requested skip for build_anndata but checkpoint missing → re-running")
        if progress_callback: progress_callback("Core pipeline modules")
        try:
            adata1, adata2, final_data = build_anndata(
                spatial_df,
                cell_df,
                nucleus_df,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
                out_dir=out_dir,
                fingerprint_context=build_fingerprint_context,
            )
            # Optional: prune final_data columns to reduce redundancy per config
            try:
                _cfg = load_params_yaml()
                _io_cfg = (_cfg.get('io', {}) if _cfg else {})
            except Exception:
                _io_cfg = {}
            drop_embeddings = bool(_io_cfg.get('drop_embeddings', False))
            keep_cols = _io_cfg.get('final_keep_columns', None)
            if isinstance(keep_cols, list) and len(keep_cols) > 0:
                existing = [c for c in keep_cols if c in final_data.columns]
                if len(existing) > 0:
                    final_data = final_data.loc[:, existing]
            elif drop_embeddings:
                # Drop large embedding columns when requested
                drop_prefixes = ("comp_emb", "gmm_feat_", "X_", "Z_", "umap_")
                final_data = final_data[[c for c in final_data.columns if not any(str(c).startswith(p) for p in drop_prefixes)]]
            persist_build = (save_intermediate and not lite_intermediate) or resume_enabled
            if persist_build:
                try:
                    _cfg = load_params_yaml()
                    comp = str((_cfg.get('io', {}) if _cfg else {}).get('h5ad_compression', compression))
                    m6_override = bool(
                        (_cfg.get('module6', {}) if _cfg else {}).get(
                            'h5ad_compression_override',
                            True,
                        )
                    )
                except Exception:
                    comp = compression
                    m6_override = True
                _save_adata(adata1, out_dir, "module5_adata1", compression=comp)
                if adata2 is not None:
                    _save_adata(
                        adata2,
                        out_dir,
                        "module6_adata2",
                        compression=comp if m6_override else "lzf",
                    )
            if resume_enabled or (save_intermediate and not lite_intermediate):
                # Resume needs the transcript-to-MetaSpot/MetaDomain ownership
                # table even in lite mode. Force one compressed Parquet rather
                # than silently making an annotate-only resume impossible.
                _save_dataframe(
                    final_data,
                    out_dir,
                    "module5_final_data",
                    force_parquet=True,
                )
            _save_state(out_dir, "build_anndata", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps, "modules_completed": True})
            logger.info("Stage build_anndata done: adata1_obs=%d adata2_obs=%s final_rows=%d", adata1.n_obs, (adata2.n_obs if adata2 is not None else None), len(final_data))
        except Exception:
            logger.exception("Stage build_anndata failed")
            raise
    else:
        logger.info("Resume: skipping build_anndata (checkpoint available)")
        if build_cache is None:
            raise RuntimeError("Missing 'build_anndata' checkpoint; cannot skip this stage.")
        adata1, adata2, cached_final = build_cache
        if cached_final is None:
            raise RuntimeError("Missing 'module5_final_data' checkpoint; cannot skip 'build_anndata' stage.")
        final_data = cached_final

    # ==== Stage 4: Annotation (optional/configurable) ====
    ann_method = (annotation_method or "evidence").lower()
    if bool(enable_annotation) and ann_method not in {"evidence", "none"}:
        raise ValueError(
            f"Unsupported annotation method {ann_method!r}; expected "
            "'evidence', or explicitly disable annotation with method='none'."
        )
    if bool(enable_annotation) and ann_method != "none" and adata2 is None:
        raise RuntimeError(
            "Annotation is enabled but MetaDomain AnnData (adata2) is missing."
        )
    run_annotation = bool(enable_annotation) and (ann_method != "none") and adata2 is not None
    annotation_cache = None
    annotation_cache_ready = False
    spot_annotation = None  # 低内存路径下每个 MetaSpot 一行的注释表；其余情况注释列已在 final_data 里
    if run_annotation and resume_enabled and stage_to_idx["annotate"] < start_idx:
        annotation_cache = _resume_annotation_outputs(out_dir)
        annotation_cache_ready = bool(
            annotation_cache
            and annotation_cache[0] is not None
            and annotation_cache[2] is not None
        )
    if run_annotation and _should_run("annotate", annotation_cache_ready):
        # 低内存路径：完整转录本表已在 module5 检查点里，内存只留注释和细胞表要用的列，写结果时再分批读回
        module5_parquet = os.path.join(_intermediate_dir(out_dir), "module5_final_data.parquet")
        low_memory_annotation = (
            final_format == "parquet" and not write_csv_copy and lite_intermediate
            and not bool((((load_params_yaml() or {}).get("io", {}) or {}).get("write_partitioned_dataset", False)))
            and os.path.isfile(module5_parquet)
        )
        if low_memory_annotation:
            build_cache = raw_inputs_cache = preproc_cache = None
            spatial_df = cell_df = nucleus_df = None
            final_data = final_data[[c for c in TRANSCRIPT_WORKING_COLS if c in final_data.columns]]
        final_data = compact_string_columns(final_data)
        if resume_enabled and stage_to_idx["annotate"] < start_idx and not annotation_cache_ready:
            logger.info("Resume requested skip for annotate but checkpoint missing → re-running")
        if progress_callback: progress_callback(f"Annotation ({ann_method})")
        try:
            adata1, final_data, spot_annotation = _annotate_with_evidence(
                adata1, final_data, gold_data, out_dir, keep_only_final_label=low_memory_annotation
            )
            required_annotation_columns = {"subcellular_annotation"} if low_memory_annotation else {
                "subcellular_annotation",
                "subcellular_annotation_domain",
            }
            missing_final_columns = sorted(
                required_annotation_columns.difference(final_data.columns)
            )
            missing_metaspot_columns = sorted(
                required_annotation_columns.difference(adata1.obs.columns)
            )
            if missing_final_columns or missing_metaspot_columns:
                raise RuntimeError(
                    "Annotation completed without required production columns; "
                    f"final_data_missing={missing_final_columns}, "
                    f"adata1_missing={missing_metaspot_columns}"
                )
            # A lite run suppresses the annotated transcript checkpoint. Saving
            # only annotated H5AD files cannot satisfy
            # ``_resume_annotation_outputs`` and merely duplicates the final
            # H5AD write, so keep this checkpoint atomic as a complete set.
            persist_annotation = save_intermediate and not lite_intermediate
            if persist_annotation:
                _save_adata(adata1, out_dir, "annotated_adata1")
                if adata2 is not None:
                    _save_adata(adata2, out_dir, "annotated_adata2")
                _save_dataframe(final_data, out_dir, "annotated_final_csv")
            elif save_intermediate:
                logger.info(
                    "Lite intermediate mode: skipping incomplete annotation "
                    "checkpoint; canonical outputs will be written once"
                )
            _save_state(out_dir, "annotate", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
            logger.info("Stage annotate done: final_rows=%d", len(final_data))
        except Exception as exc:
            logger.exception(
                "Stage annotate failed; refusing to write a successful "
                "production output"
            )
            raise RuntimeError(
                f"Annotation stage failed: {exc}"
            ) from exc
    elif run_annotation and annotation_cache_ready and annotation_cache is not None:
        logger.info("Resume: skipping annotate (checkpoint available)")
        ann_adata1, ann_adata2, ann_final = annotation_cache
        if ann_adata1 is not None:
            adata1 = ann_adata1
        if ann_adata2 is not None:
            adata2 = ann_adata2
        final_data = ann_final
        required_annotation_columns = {
            "subcellular_annotation",
            "subcellular_annotation_domain",
        }
        missing_cached_columns = sorted(
            required_annotation_columns.difference(final_data.columns)
        )
        if missing_cached_columns:
            raise RuntimeError(
                "Cached annotation output is incomplete; missing "
                f"{missing_cached_columns}. Rerun annotation."
            )
        missing_cached_metaspot_columns = sorted(
            required_annotation_columns.difference(adata1.obs.columns)
        )
        if missing_cached_metaspot_columns:
            raise RuntimeError(
                "Cached MetaSpot annotation output is incomplete; missing "
                f"{missing_cached_metaspot_columns}. Rerun annotation."
            )
    else:
        logger.info(
            "Annotation disabled explicitly | enable=%s | method=%s",
            bool(enable_annotation),
            ann_method,
        )

    if adata1 is None or final_data is None:
        raise RuntimeError("Cannot write outputs: missing 'adata1' or 'final_data'. Please check prior stages or checkpoints.")
    adata1 = cast(ad.AnnData, adata1)
    final_data = cast(pd.DataFrame, final_data)

    # Step 3: Write outputs
    adata1_path = os.path.join(out_dir, "adata1.h5ad")
    adata2_path = os.path.join(out_dir, "adata2.h5ad")
    final_parquet_path = os.path.join(out_dir, "final_data.parquet")
    final_csv_path = os.path.join(out_dir, "final_data.csv")
    final_dataset_path = os.path.join(out_dir, "final_data_dataset")

    # Optionally resume directly to writing outputs
    if resume_from == "write_outputs":
        pass

    # Write files

    if progress_callback: progress_callback("Writing outputs")
    written_outputs: Dict[str, Optional[str]] = {
        "adata1": None,
        "adata2": None,
        "final_parquet": None,
        "final_csv": None,
        "final_dataset": None,
        "stale_final_parquet": None,
    }
    try:
        # Compatibility for anndata<0.11 + pandas StringArray/categorical writes.
        # First sanitize obs columns, then opt-in to nullable-string writing when available.
        adata1 = _sanitize_obs_for_h5ad(adata1)
        try:
            _write_cfg = load_params_yaml() or {}
            _drop_final_h5ad_embeddings = bool(
                (_write_cfg.get("io", {}) or {}).get(
                    "drop_embeddings",
                    False,
                )
            )
        except Exception:
            _drop_final_h5ad_embeddings = False
        try:
            import anndata as _ad
            _ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        with _compact_final_adata1_for_write(
            adata1,
            enabled=_drop_final_h5ad_embeddings,
        ):
            _atomic_write_adata(
                adata1,
                adata1_path,
                compression=compression,
            )
        written_outputs["adata1"] = adata1_path
        if adata2 is not None:
            adata2 = _sanitize_obs_for_h5ad(adata2)
            _atomic_write_adata(
                adata2,
                adata2_path,
                compression=compression,
            )
            written_outputs["adata2"] = adata2_path
        if final_format == "csv":
            _atomic_dataframe_write(
                final_data,
                final_csv_path,
                file_format="csv",
            )
            written_outputs["final_csv"] = final_csv_path
            try:
                _atomic_dataframe_write(
                    final_data,
                    final_parquet_path,
                    file_format="parquet",
                )
                written_outputs["final_parquet"] = final_parquet_path
            except Exception as exc:
                stale_path = _quarantine_stale_file(final_parquet_path)
                written_outputs["stale_final_parquet"] = stale_path
                logger.warning(
                    "Optional Parquet copy was not written; CSV production "
                    "output remains valid | stale_parquet=%s | error=%s",
                    stale_path,
                    exc,
                )
        else:  # parquet chosen
            try:
                if spot_annotation is not None:
                    write_annotated_transcripts(
                        module5_parquet, spot_annotation, final_parquet_path,
                        {c: final_data[c].cat.categories for c in ("feature_name", "meta-domain")
                         if c in final_data.columns and isinstance(final_data[c].dtype, pd.CategoricalDtype)},
                    )
                else:
                    _atomic_dataframe_write(
                        final_data,
                        final_parquet_path,
                        file_format="parquet",
                    )
                written_outputs["final_parquet"] = final_parquet_path
                if write_csv_copy:
                    try:
                        _atomic_dataframe_write(
                            final_data,
                            final_csv_path,
                            file_format="csv",
                        )
                        written_outputs["final_csv"] = final_csv_path
                    except Exception as exc:
                        logger.warning(
                            "Optional CSV copy was not written; Parquet "
                            "production output remains valid | error=%s",
                            exc,
                        )
                elif os.path.isfile(final_csv_path):
                    # Do not leave a stale CSV from an older run beside the
                    # newly written canonical Parquet output.
                    os.unlink(final_csv_path)
                    logger.info(
                        "Removed stale optional CSV output: %s",
                        final_csv_path,
                    )
            except Exception as exc:
                logger.warning(
                    "Parquet production write failed; falling back to CSV | "
                    "error=%s",
                    exc,
                    exc_info=True,
                )
                _atomic_dataframe_write(
                    final_data,
                    final_csv_path,
                    file_format="csv",
                )
                written_outputs["final_csv"] = final_csv_path
                stale_path = _quarantine_stale_file(final_parquet_path)
                written_outputs["stale_final_parquet"] = stale_path
                if stale_path:
                    logger.warning(
                        "Quarantined stale Parquet output: %s",
                        stale_path,
                    )
        execution_plan = adata1.uns.get("execution_plan", {})
        write_partitioned_dataset = bool(
            io_cfg.get("write_partitioned_dataset", False)
        )
        if (
            bool(execution_plan.get("partitioned_annotation", False))
            and write_partitioned_dataset
        ):
            partition_manifest = _atomic_partitioned_parquet_write(
                final_data,
                final_dataset_path,
                target_rows_per_part=int(
                    execution_plan.get("partition_rows", 1_000_000)
                ),
            )
            written_outputs["final_dataset"] = final_dataset_path
            logger.info(
                "Streaming final dataset written | path=%s | parts=%d",
                final_dataset_path,
                int(partition_manifest["n_parts"]),
            )
        _save_state(out_dir, "write_outputs", extra={"fingerprints": cur_fps, "module_fingerprints": cur_mod_fps})
        logger.info(
            "Stage write_outputs done | adata1=%s | adata2=%s | "
            "final_parquet=%s | final_csv=%s | final_dataset=%s",
            written_outputs["adata1"],
            written_outputs["adata2"],
            written_outputs["final_parquet"],
            written_outputs["final_csv"],
            written_outputs["final_dataset"],
        )
    except Exception:
        logger.exception("Stage write_outputs failed")
        raise

    if run_annotation and bool(((load_params_yaml() or {}).get("io", {}) or {}).get("cell_structure_features", False)):
        written_outputs["cell_structure_features"] = _write_cell_structure_features(adata1, final_data, gold_data, out_dir)

    if _io_should_write_figures():
        _write_run_figures_from_outputs(out_dir, adata1, final_data, written_outputs.get("cell_structure_features"), gold_data)

    counts = {
        "n_transcripts": int(len(final_data)),
        "n_cells": (
            int(final_data["cell_id"].nunique())
            if "cell_id" in final_data.columns
            else None
        ),
        "n_metaspots": int(adata1.n_obs),
        "n_metadomains": (
            int(adata2.n_obs) if adata2 is not None else None
        ),
    }
    output_guide_path = _write_output_guide(
        out_dir,
        counts=counts,
        annotation_enabled=bool(run_annotation),
        primary_data_product=str(
            (
                written_outputs["final_csv"]
                if final_format == "csv"
                else written_outputs["final_parquet"]
            )
            or written_outputs["final_csv"]
        ),
    )
    written_outputs["output_guide"] = output_guide_path
    logger.info(
        "Pipeline finished successfully | transcripts=%s | cells=%s | "
        "metaspots=%s | metadomains=%s",
        counts["n_transcripts"],
        counts["n_cells"],
        counts["n_metaspots"],
        counts["n_metadomains"],
    )
    return {**written_outputs, "counts": counts}

@functools.wraps(_run_pipeline_impl)
def run_pipeline(*args, **kwargs) -> dict:
    """Run one isolated pipeline invocation.

    Module helpers use temporary process-global I/O switches while a run is
    active. Always restore them so failed runs and notebook dry-runs cannot
    change the behavior of a later invocation in the same Python process.
    """
    previous = io_switches()
    try:
        return _run_pipeline_impl(*args, **kwargs)
    finally:
        set_io_switches(*previous)

def _reference_marker_sets(gene_names, gold_data) -> Dict[str, List[str]]:
    """注释所用的参考 marker 集合（结构 → 基因）；测到足够线粒体编码 RNA 时 OMM 用它们，与注释一致。"""
    from .evidence_annotation import _MIN_MITO_ENCODED, marker_sets_from_reference

    annotation_cfg = resolve_annotation_configuration(load_params_yaml() or {})
    markers = marker_sets_from_reference(_load_reference_scores(gold_data, annotation_cfg),
                                         float(annotation_cfg["evidence_marker_margin"]))
    mito = sorted({g for g in pd.Index(gene_names).astype(str).str.upper() if g.startswith("MT-")})
    if len(mito) >= _MIN_MITO_ENCODED:
        markers["OMM"] = mito
    return markers


def _write_run_figures_from_outputs(out_dir: str, adata1: ad.AnnData, final_df: pd.DataFrame,
                                    cell_features_path: Optional[str], gold_data=None) -> None:
    """运行结束时把诊断图写到 out_dir/figures/；边界从预处理断点读，细胞 × 结构表从刚写出的文件读。"""
    from .run_figures import write_run_figures

    def read_optional(path):
        try:
            return pd.read_parquet(path) if path and os.path.isfile(path) else None
        except Exception as exc:
            _logging.getLogger("cellscoper").warning("Diagnostic figures: could not read %s: %s", path, exc)
            return None

    write_run_figures(
        out_dir, adata1, final_df,
        cell_features=read_optional(cell_features_path),
        cell_bounds=read_optional(_checkpoint_path(out_dir, "cell_preproc")),
        nucleus_bounds=read_optional(_checkpoint_path(out_dir, "nucleus_preproc")),
        markers=_reference_marker_sets(pd.unique(final_df["feature_name"]), gold_data),
    )
    _logging.getLogger("cellscoper").info("Diagnostic figures written | path=%s", os.path.join(out_dir, FIGURES_DIR))


def _write_cell_structure_features(
    adata1: ad.AnnData,
    final_df: pd.DataFrame,
    gold_data: Optional[Union[str, pd.DataFrame]],
    out_dir: str,
) -> Optional[str]:
    """Cell x structure table (io.cell_structure_features): label shares and pair co-localisation of every
    structure label and every MetaDomain. Written next to final_data."""
    from .cell_features import ge_structure_weights, metadomain_gene_weights, structure_feature_table

    log = _logging.getLogger("cellscoper")
    ge = adata1.uns.get("gene_embedding")
    if not ge or "vectors" not in ge:
        log.warning("io.cell_structure_features needs the spatial gene embedding; skipped")
        return None
    markers = _reference_marker_sets(pd.unique(final_df["feature_name"]), gold_data)
    structure_w = ge_structure_weights(np.asarray(ge["vectors"]), np.asarray(ge["genes"]).astype(str), markers)
    radius = float(np.asarray(ge["neighbour_radii"])[1])  # 基因嵌入的中间尺度，由数据定
    disc = adata1.uns.get("metadomain_discovery") or {}
    domain_w = None
    if "meta-domain" in final_df.columns and len(np.asarray(disc.get("gene_profiles", ()))):
        domain_w = metadomain_gene_weights(np.asarray(disc["gene_profiles"]), np.asarray(disc["gene_names"]).astype(str),
                                           [f"MD{int(k)}" for k in np.asarray(disc["metadomain"])])
    table = structure_feature_table(final_df, structure_weights=structure_w, radius=radius,
                                    domain_col="meta-domain" if domain_w is not None else None,
                                    domain_weights=domain_w, n_jobs=max(1, _preferred_n_jobs()))
    path = os.path.join(out_dir, "cell_structure_features.parquet")
    _atomic_dataframe_write(table, path, file_format="parquet")
    log.info("Cell x structure features written | rows=%d | features=%d | radius=%.3f | path=%s",
             len(table), table["feature"].nunique(), radius, path)
    return path

def _annotate_with_evidence(
    adata1: ad.AnnData,
    final_df: pd.DataFrame,
    gold_data: Optional[Union[str, pd.DataFrame]],
    out_dir: Optional[str],
    keep_only_final_label: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame, Optional[pd.DataFrame]]:
    """Run the evidence annotation and write its columns and report.

    Returns (adata1, final_df, spot table). ``keep_only_final_label``: final_df gets only subcellular_annotation (what the
    cell table needs) and the MetaSpot table is returned for ``write_annotated_transcripts``; otherwise every annotation
    column goes on final_df and the third value is None."""
    import json

    from .evidence_annotation import run_evidence_annotation

    log = _logging.getLogger("cellscoper")
    annotation_cfg = resolve_annotation_configuration(load_params_yaml() or {})
    reference = _load_reference_scores(gold_data, annotation_cfg)
    if GENE_EMBEDDING_REP not in adata1.obsm:  # MetaDomains came from another representation
        adata1.obsm[GENE_EMBEDDING_REP] = _compute_gene_embedding(
            final_df, adata1.obs_names, (load_params_yaml() or {}).get("module5", {}) or {}, adata1.uns
        )
    axis_inputs = {
        "spot_names": adata1.obs_names.astype(str).to_numpy(),
        "ge": adata1.uns.get("gene_embedding"),
        "ge_block": adata1.obsm[GENE_EMBEDDING_REP],
        "lens": adata1.uns.get("lens"),
        "z": adata1.obsm[LENS_REP] if LENS_REP in adata1.obsm else None,
        "radial": adata1.obs["d_cell_norm_mean"].to_numpy() if "d_cell_norm_mean" in adata1.obs else None,
    }
    table, report = run_evidence_annotation(final_df, reference, annotation_cfg, axis_inputs)
    table = table.set_index("metaspot")

    # MetaDomain summary: the most common primary label of its MetaSpots,
    # separately on each side of the nuclear boundary.
    first_row = ~final_df["metaspot"].duplicated().to_numpy()
    spot_domain = pd.Series(
        final_df["meta-domain"].to_numpy()[first_row],
        index=final_df["metaspot"].to_numpy()[first_row],
    ).astype(str).reindex(table.index)
    domain_key = spot_domain + "|" + table["subcellular_broad_compartment"].astype(str)
    table["subcellular_annotation_domain"] = (
        table["subcellular_annotation_validated"]
        .groupby(domain_key.to_numpy())
        .transform(lambda labels: labels.value_counts().index[0])
    )

    spot_col = final_df["metaspot"]
    rows = table.index.get_indexer(spot_col.cat.categories)[spot_col.cat.codes.to_numpy()]

    def _transcript_values(name: str, values: np.ndarray):
        # 文字标签按整数编码展开到每条转录本，不为每行存一个对象
        if name not in COMPACT_STRING_COLS:
            return values[rows]
        categorical = pd.Categorical(values)
        return pd.Categorical.from_codes(categorical.codes[rows], categories=categorical.categories)

    final_df = _replace_frame_columns(
        final_df,
        {name: _transcript_values(name, table[name].to_numpy())
         for name in (("subcellular_annotation",) if keep_only_final_label else table.columns)},
    )
    obs_rows = table.index.get_indexer(adata1.obs_names.astype(str))
    if (obs_rows < 0).any():
        raise RuntimeError(
            "Evidence annotation: adata1 MetaSpots are missing from final_data"
        )
    _replace_obs_columns(
        adata1, {name: table[name].to_numpy()[obs_rows] for name in table.columns}
    )

    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (np.integer, np.bool_)):
            return value.item()
        return value

    report = _json_safe(report)
    adata1.uns["evidence_annotation"] = json.dumps(report)
    if out_dir:
        atomic_write_json(
            os.path.join(out_dir, "evidence_annotation_report.json"), report
        )
    resolution = report["resolution"]
    log.info(
        "Evidence annotation | resolution_spearman=%s | supported=%s | "
        "placed without held-out support=%s",
        resolution.get("split_half_spearman"),
        report["supported_structures"] or "none",
        report["unsupported_structures"] or "none",
    )
    for name, entry in report["structures"].items():
        level = entry.get("level")
        stats = (entry.get("levels") or {}).get(level, {})
        log.info(
            "  %-12s supported=%s level=%s held-out marker enrichment=%s "
            "(95%% CI %s-%s) units=%s markers=%s",
            name,
            entry.get("supported"),
            level,
            stats.get("marker_enrichment"),
            stats.get("ci_low"),
            stats.get("ci_high"),
            entry.get("units_called"),
            entry.get("marker_genes_used"),
        )
    return adata1, final_df, (table if keep_only_final_label else None)
