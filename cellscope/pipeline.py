from typing import Optional, Literal, Tuple, List, Callable, cast, Dict, Any, Union, Sequence, Mapping
from contextlib import contextmanager
import functools
import os
import re
import math
import hashlib
import logging as _logging
import tempfile
import time as _time
import warnings as _warnings
from collections import defaultdict, Counter

# Numba and Matplotlib otherwise try to cache beside read-only site-packages or
# under a non-writable home in containers/HPC jobs. Keep caches process-local
# and writable without changing numerical behavior.
_runtime_cache_root = os.path.join(tempfile.gettempdir(), "cellscope-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_runtime_cache_root, "numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_runtime_cache_root, "matplotlib"))

# Per-cell spatial graphs may contain several legitimate islands.  sklearn
# connects those islands internally before constrained Ward clustering and
# otherwise emits this warning once per affected cell.  The connectivity
# contract is enforced again below, so keep only this exact repeated notice
# out of user-facing progress output.
def _quiet_expected_disconnected_connectivity_warning() -> None:
    _warnings.filterwarnings(
        "ignore",
        message=(
            r"the number of connected components of the connectivity matrix is "
            r"\d+ > 1\. Completing it to avoid stopping the tree early\."
        ),
        category=UserWarning,
        module=r"sklearn\.cluster\._agglomerative",
    )


_quiet_expected_disconnected_connectivity_warning()

# Optional early config load so runtime.blas_threads / runtime.n_jobs in params.yaml
# can cap BLAS/OMP threads before importing heavy numeric libs.
try:
    from .config import load_params_yaml as _load_params_yaml_threads
except Exception:  # pragma: no cover - defensive for import edge cases
    _load_params_yaml_threads = None

def _coerce_pos_int(v: object) -> Optional[int]:
    try:
        iv = int(v)  # type: ignore[arg-type]
        return iv if iv > 0 else None
    except Exception:
        return None

# Configure BLAS/OMP threads before importing numpy/scipy.
# Priority: params.yaml runtime.blas_threads / runtime.n_jobs > fallback heuristic;
# inherited BLAS environment values are deliberately overridden for reproducibility.
_runtime_cfg_threads = {}
try:
    if _load_params_yaml_threads:
        _runtime_cfg_threads = (_load_params_yaml_threads() or {}).get("runtime", {}) or {}
except Exception:
    _runtime_cfg_threads = {}

_cpu = os.cpu_count() or 8
_fallback_threads = max(1, min(32, _cpu // 2 or 1))
_yaml_threads = _coerce_pos_int(_runtime_cfg_threads.get("blas_threads")) or _coerce_pos_int(_runtime_cfg_threads.get("n_jobs"))
_outer_jobs = (
    _coerce_pos_int(os.environ.get("CELLSCOPE_N_JOBS"))
    or _coerce_pos_int(_runtime_cfg_threads.get("n_jobs"))
    or 1
)
_threaded_outer_parallelism = (
    str(
        os.environ.get("CELLSCOPE_JOBLIB_BACKEND")
        or _runtime_cfg_threads.get("joblib_backend", "")
    ).strip().lower()
    == "threading"
    and _outer_jobs > 1
)
# The bundled OpenBLAS was built with a maximum thread table of 64.  Above
# that value it may warn at import time and subsequently segfault under BLAS
# load, so keep both the initial import environment and runtime limits within
# the compiled capability.
_MAX_SAFE_BLAS_THREADS = 64
_threads_default = str(
    1
    if _threaded_outer_parallelism
    else min(_yaml_threads or _fallback_threads, _MAX_SAFE_BLAS_THREADS)
)
_numba_safe_threads = (
    max(1, _MAX_SAFE_BLAS_THREADS // _outer_jobs)
    if _threaded_outer_parallelism
    else _MAX_SAFE_BLAS_THREADS
)
_numba_threads_default = str(
    min(
        _coerce_pos_int(_runtime_cfg_threads.get("numba_threads"))
        or (1 if _threaded_outer_parallelism else (_yaml_threads or _fallback_threads)),
        _numba_safe_threads,
    )
)
_scanpy_threads_default = str(
    min(
        _coerce_pos_int(os.environ.get("SCANPY_N_JOBS"))
        or _coerce_pos_int(_runtime_cfg_threads.get("scanpy_n_jobs"))
        or 4,
        int(_numba_threads_default),
    )
)
_THREAD_ENV_DEFAULTS = {
    # These must be assigned before importing numpy/scipy.  ``setdefault``
    # previously preserved a user's inherited OPENBLAS_NUM_THREADS value even
    # when runtime.blas_threads requested a smaller safe limit; OpenBLAS then
    # initialized with an oversized thread table and could segfault later.
    "OPENBLAS_NUM_THREADS": _threads_default,
    "OMP_NUM_THREADS": _threads_default,
    "MKL_NUM_THREADS": _threads_default,
    "NUMEXPR_MAX_THREADS": _threads_default,
    "NUMEXPR_NUM_THREADS": _threads_default,
    "BLIS_NUM_THREADS": _threads_default,
    "VECLIB_MAXIMUM_THREADS": _threads_default,
    "ACCELERATE_NUM_THREADS": _threads_default,
    # Numba-backed UMAP kernels can call BLAS from their worker threads.  On
    # hosts with more cores than OpenBLAS was compiled for, leaving Numba at
    # os.cpu_count() can exhaust OpenBLAS' thread-metadata table even though
    # OPENBLAS_NUM_THREADS itself is 1.
    "NUMBA_NUM_THREADS": _numba_threads_default,
    "SCANPY_N_JOBS": _scanpy_threads_default,
}
for _k, _v in _THREAD_ENV_DEFAULTS.items():
    os.environ[_k] = str(_v)
del _THREAD_ENV_DEFAULTS, _k, _v, _cpu, _threads_default, _fallback_threads, _yaml_threads, _runtime_cfg_threads, _outer_jobs, _threaded_outer_parallelism, _numba_safe_threads, _numba_threads_default, _scanpy_threads_default

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import anndata as ad
from scipy.stats import binom, hypergeom, norm, rankdata, fisher_exact
from scipy.special import erfc
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.neighbors import KDTree
def _kdtree_query(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build KDTree and query with tuned params; returns (indices, distances).
    Points must be float32 array of shape (n, 2). k will be clamped to [1, n].
    """
    n = points.shape[0]
    if n == 0:
        return None, None
    k = max(1, min(int(k), n))
    leaf = max(20, min(64, n // 10 if n >= 100 else 20))
    # breadth_first/sort_results improve determinism; avoid dualtree for small n
    kdt = KDTree(points, leaf_size=leaf)
    try:
        dist, ind = kdt.query(points, k=k, return_distance=True, breadth_first=True, sort_results=True)
    except TypeError:
        # Fallback for older sklearn without options
        dist, ind = kdt.query(points, k=k)
    ind = np.asarray(ind)
    if ind.dtype != np.int64:
        ind = ind.astype(np.int64, copy=False)
    return ind, dist
from matplotlib.path import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin
import scipy.sparse as sp
import gc

from .plotting_style import (
    PUBLICATION_PLOT_STYLE as _PUBLICATION_PLOT_STYLE,
    SUBCELLULAR_LABEL_GROUPS as _SUBCELLULAR_LABEL_GROUPS,
    SUBCELLULAR_LABEL_ORDER as _SUBCELLULAR_LABEL_ORDER,
    SUBCELLULAR_LABEL_PALETTE as _SUBCELLULAR_LABEL_PALETTE,
)


METASPOT_COL = "metaspot"
METASPOT_LOCAL_COL = "metaspot_in_cell"
_LEGACY_METASPOT_COL = "cell_sub"
_LEGACY_METASPOT_LOCAL_COL = "cluster_in_cell"

METASPOT_GRAPH_SCHEMA_VERSION = 4
METASPOT_DGI_SCHEMA_VERSION = 4
METASPOT_DGI_KEY = "X_dgi_gcn_metaspot"
METASPOT_DGI_SAGE_KEY = "X_dgi_sage_metaspot"
METADOMAIN_DGI_KEY = "X_dgi_gcn_metadomain"
METASPOT_GRAPH_KEY = "A_metaspot_norm"
METASPOT_DMGI_EXPRESSION_KEY = "X_dmgi_expression_metaspot"
METASPOT_DMGI_SPATIAL_KEY = "X_dmgi_spatial_metaspot"


def _replace_frame_columns(
    frame: pd.DataFrame,
    columns: Mapping[str, Any],
) -> pd.DataFrame:
    """Return ``frame`` with many columns replaced in one column concat.

    Repeated ``frame[name] = values`` calls create one pandas block at a time.
    Annotation writes more than one hundred audit columns, so that pattern
    fragments both ``AnnData.obs`` and transcript tables and makes every later
    insertion emit ``PerformanceWarning``. Building the update frame first
    keeps dtype inference (including categoricals and nullable booleans) while
    avoiding the repeated insertion path.
    """
    if not columns:
        return frame

    updates: Dict[str, Any] = {}
    for name, values in columns.items():
        if isinstance(values, pd.Series):
            updates[str(name)] = values.reindex(frame.index)
        else:
            updates[str(name)] = values
    update_frame = pd.DataFrame(updates, index=frame.index)
    base = frame.drop(columns=list(update_frame.columns), errors="ignore")
    return pd.concat([base, update_frame], axis=1, copy=False)


def _replace_obs_columns(
    adata: ad.AnnData,
    columns: Mapping[str, Any],
) -> None:
    """Bulk-replace AnnData observation columns without fragmenting obs."""
    if columns:
        adata.obs = _replace_frame_columns(adata.obs, columns)


def _obs_names_checksum(obs_names: Sequence[object]) -> str:
    """Return an order-sensitive digest for an AnnData observation index."""
    digest = hashlib.sha256()
    for value in obs_names:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _membership_checksum(
    obs_names: Sequence[object],
    memberships: Sequence[object],
) -> str:
    """Return an order-sensitive digest of observation-to-group membership."""
    digest = hashlib.sha256()
    for obs_name, membership in zip(obs_names, memberships):
        digest.update(str(obs_name).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(membership).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _migrate_legacy_metaspot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical MetaSpot columns when loading old checkpoints."""
    rename = {}
    if METASPOT_COL not in df.columns and _LEGACY_METASPOT_COL in df.columns:
        rename[_LEGACY_METASPOT_COL] = METASPOT_COL
    if (
        METASPOT_LOCAL_COL not in df.columns
        and _LEGACY_METASPOT_LOCAL_COL in df.columns
    ):
        rename[_LEGACY_METASPOT_LOCAL_COL] = METASPOT_LOCAL_COL
    if rename:
        df = df.rename(columns=rename)
    stale = [
        column
        for column in (_LEGACY_METASPOT_COL, _LEGACY_METASPOT_LOCAL_COL)
        if column in df.columns
    ]
    if stale:
        df = df.drop(columns=stale)
    return df


def _migrate_legacy_metaspot_obs(adata: ad.AnnData) -> ad.AnnData:
    """Normalize historical MetaSpot columns in an AnnData checkpoint."""
    rename = {}
    if METASPOT_COL not in adata.obs.columns and _LEGACY_METASPOT_COL in adata.obs.columns:
        rename[_LEGACY_METASPOT_COL] = METASPOT_COL
    if (
        METASPOT_LOCAL_COL not in adata.obs.columns
        and _LEGACY_METASPOT_LOCAL_COL in adata.obs.columns
    ):
        rename[_LEGACY_METASPOT_LOCAL_COL] = METASPOT_LOCAL_COL
    if rename:
        adata.obs = adata.obs.rename(columns=rename)
    stale = [
        column
        for column in (_LEGACY_METASPOT_COL, _LEGACY_METASPOT_LOCAL_COL)
        if column in adata.obs.columns
    ]
    if stale:
        adata.obs = adata.obs.drop(columns=stale)
    return adata


class _LocalCompositionModel:
    """Compact local-expression transformer used one cell at a time."""

    def __init__(
        self,
        gene2idx: Dict[str, int],
        transformer: Any,
        idf_vec: np.ndarray,
        n_components: int,
        knn_k: int,
        weight_mode: str,
        tf_norm: bool,
        use_tfidf: bool,
    ):
        self.gene2idx = gene2idx
        self.transformer = transformer
        self.idf_vec = np.asarray(idf_vec, dtype=np.float32)
        self.n_components = int(n_components)
        self.knn_k = int(knn_k)
        self.weight_mode = str(weight_mode)
        self.tf_norm = bool(tf_norm)
        self.use_tfidf = bool(use_tfidf)

    @property
    def feature_names(self) -> List[str]:
        return [f"comp_emb{i}" for i in range(self.n_components)]

    def transform_cell(self, dfc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(dfc)
        if n == 0:
            return (
                np.zeros((0, self.n_components), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        points = dfc[["x_location", "y_location"]].to_numpy(np.float32, copy=False)
        ind, dist = _kdtree_query(points, k=min(self.knn_k, n))
        gene_idx = (
            dfc["feature_name"]
            .astype(str)
            .map(self.gene2idx)
            .fillna(-1)
            .to_numpy(dtype=np.int64, copy=False)
        )
        k = ind.shape[1]

        if self.weight_mode == "gaussian":
            radius = dist[:, -1].astype(np.float32) + 1e-6
            weights = np.exp(-((dist / radius[:, None]) ** 2)).astype(np.float32)
        elif self.weight_mode == "inv":
            weights = (1.0 / (dist + 1e-3)).astype(np.float32)
        else:
            weights = np.ones_like(dist, dtype=np.float32)

        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        neighbors = ind.ravel().astype(np.int64, copy=False)
        cols = gene_idx[neighbors]
        values = weights.ravel()
        valid = (cols >= 0) & (cols < len(self.gene2idx))
        bow = sp.coo_matrix(
            (values[valid], (rows[valid], cols[valid])),
            shape=(n, len(self.gene2idx)),
            dtype=np.float32,
        ).tocsr()

        if self.tf_norm:
            row_sum = np.asarray(bow.sum(axis=1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            bow = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        if self.use_tfidf:
            bow = bow.multiply(self.idf_vec[np.newaxis, :]).tocsr()

        z = self.transformer.transform(bow)
        if sp.issparse(z):
            z = z.toarray()
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        if z.shape[1] < self.n_components:
            z = np.pad(z, ((0, 0), (0, self.n_components - z.shape[1])))
        elif z.shape[1] > self.n_components:
            z = z[:, :self.n_components]

        radius = dist[:, -1].astype(np.float32) + 1e-3
        density = (k / (np.pi * radius * radius)).astype(np.float32)
        return z, density


# Keep Scanpy/PyNNDescent within Numba's import-time thread ceiling.
try:
    import scanpy as sc
    sc.settings.n_jobs = max(
        1,
        min(
            int(os.environ.get("SCANPY_N_JOBS", "4")),
            int(os.environ["NUMBA_NUM_THREADS"]),
        ),
    )
except Exception:
    pass

def _progress_iter(iterator, desc: str = "", total: Optional[int] = None, show: bool = False):
    """Terminal-friendly progress iterator with stage messages and ETA.
    - TTY: Rich single-line bar with ETA; prints start/done summaries.
    - Non-TTY: milestone prints at 0/25/50/75/100% with ETA + done summary.
    """
    if not show:
        return iterator
    import time as _time
    _desc = desc or "Working"
    # Try infer total if not provided
    if total is None:
        try:
            total = len(iterator)  # may fail for generators/groupby
        except Exception:
            total = None
    # Attempt Rich progress (best single-line UX)
    try:
        from rich.console import Console as _Console
        from rich.progress import (
            Progress as _Progress,
            SpinnerColumn as _Spinner,
            BarColumn as _Bar,
            TextColumn as _Text,
            TimeElapsedColumn as _TimeElapsed,
            TimeRemainingColumn as _TimeRemain,
            TaskProgressColumn as _TaskProg,
            MofNCompleteColumn as _MofN,
        )
        _con = _Console(stderr=True)
        if _con.is_terminal:  # TTY only
            start_t = _time.perf_counter()
            _con.print(f">> Start: {_desc}")
            if total is None:
                columns = [
                    _Spinner(),
                    _Text(" {task.description}"),
                    _Text("  elapsed:"), _TimeElapsed(),
                ]
                prog = _Progress(*columns, transient=True, console=_con)
                def _gen_rich_spin():
                    task = prog.add_task(_desc, total=None)
                    with prog:
                        c = 0
                        for item in iterator:
                            c += 1
                            prog.advance(task, 1)
                            yield item
                    elapsed = _time.perf_counter() - start_t
                    rate = f"{c/elapsed:,.1f}/s" if elapsed > 0 else "n/a"
                    _con.print(f"<< Done: {_desc} in {elapsed:0.2f}s (items={c}, rate={rate})")
                return _gen_rich_spin()
            else:
                columns = [
                    _Text("{task.description}: "),
                    _Bar(),
                    _TaskProg(),
                    _Text("  "), _MofN(),
                    _Text("  elapsed:"), _TimeElapsed(),
                    _Text("  ETA:"), _TimeRemain(),
                ]
                prog = _Progress(*columns, transient=True, console=_con)
                def _gen_rich_bar():
                    task = prog.add_task(_desc, total=total)
                    with prog:
                        c = 0
                        for item in iterator:
                            c += 1
                            prog.update(task, completed=c)
                            yield item
                    elapsed = _time.perf_counter() - start_t
                    rate = f"{c/elapsed:,.1f}/s" if elapsed > 0 else "n/a"
                    _con.print(f"<< Done: {_desc} in {elapsed:0.2f}s ({c}/{total}, rate={rate})")
                return _gen_rich_bar()
    except Exception:
        pass
    # Fallback: milestone-only prints with ETA
    milestones = [0, 25, 50, 75, 100] if (isinstance(total, int) and total >= 20) else [0]
    def _eta_str(done: int, tot: int, elapsed: float) -> str:
        if tot and done > 0:
            frac = done / tot
            remain = max((elapsed / frac) - elapsed, 0.0)
            # format mm:ss
            mm = int(remain // 60); ss = int(remain % 60)
            return f"ETA {mm:02d}:{ss:02d}"
        return "ETA --:--"
    def _gen_milestone():
        start = _time.perf_counter(); cnt = 0
        if total:
            print(f">> Start: {_desc} (0/{total})")
            print(f"{_desc}: 0% (0/{total}) {_eta_str(0, total, 0.0)}")
            if milestones and milestones[0] == 0:
                milestones.pop(0)
        else:
            print(f">> Start: {_desc}")
        for item in iterator:
            cnt += 1
            if total:
                perc = int(cnt * 100 / max(total, 1))
                while len(milestones) and perc >= milestones[0]:
                    m = milestones.pop(0)
                    eta = _eta_str(cnt, total, _time.perf_counter() - start)
                    print(f"{_desc}: {m}% ({cnt}/{total}) {eta}")
            yield item
        elapsed = _time.perf_counter() - start
        rate = f"{(cnt/elapsed):,.1f}/s" if elapsed > 0 else "n/a"
        if total:
            print(f"<< Done: {_desc} in {elapsed:0.2f}s ({cnt}/{total}, rate={rate})")
        else:
            print(f"<< Done: {_desc} in {elapsed:0.2f}s (items={cnt}, rate={rate})")
    return _gen_milestone()
from .io import (
    read_spatial_transcripts,
    read_boundaries,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
)
from .logging_utils import setup_logger
from .config import (
    dgi_unit_from_config,
    fingerprint_modules_set,
    fingerprint_stage,
    load_params_yaml,
    resolve_annotation_configuration,
)
from .provenance import (
    atomic_write_json,
    build_input_provenance,
    bundled_reference_provenance,
    sha256_mapping,
    source_tree_provenance,
)
from .resource_paths import bundled_resource_path


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


def _load_df_checkpoint_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return load_checkpoint(path)
    except Exception:
        return None


def _resume_raw_inputs(out_dir: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    cp_spatial = _checkpoint_path(out_dir, "spatial")
    cp_cell = _checkpoint_path(out_dir, "cell_boundaries")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_boundaries")
    spatial_df = _load_df_checkpoint_if_exists(cp_spatial)
    cell_df = _load_df_checkpoint_if_exists(cp_cell)
    nuc_df = _load_df_checkpoint_if_exists(cp_nuc)
    if spatial_df is None or cell_df is None or nuc_df is None:
        return None
    return spatial_df, cell_df, nuc_df


def _resume_preprocessed_inputs(out_dir: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    cp_spatial = _checkpoint_path(out_dir, "spatial_preproc")
    cp_cell = _checkpoint_path(out_dir, "cell_preproc")
    cp_nuc = _checkpoint_path(out_dir, "nucleus_preproc")
    spatial_df = _load_df_checkpoint_if_exists(cp_spatial)
    cell_df = _load_df_checkpoint_if_exists(cp_cell)
    nuc_df = _load_df_checkpoint_if_exists(cp_nuc)
    if spatial_df is None or cell_df is None or nuc_df is None:
        return None
    return spatial_df, cell_df, nuc_df


def _load_dataframe_checkpoint(base_dir: str, stem: str) -> Optional[pd.DataFrame]:
    csv_path = os.path.join(base_dir, f"{stem}.csv")
    pq_path = os.path.join(base_dir, f"{stem}.parquet")
    try:
        if os.path.exists(pq_path):
            return _migrate_legacy_metaspot_columns(pd.read_parquet(pq_path))
        if os.path.exists(csv_path):
            return _migrate_legacy_metaspot_columns(pd.read_csv(csv_path))
    except Exception:
        return None
    return None


def _module3_membership_manifest_path(out_dir: str) -> str:
    return os.path.join(
        _intermediate_dir(out_dir),
        "module3_metaspot_membership.manifest.json",
    )


def _save_module3_membership_checkpoint(
    point_df: pd.DataFrame,
    out_dir: str,
    *,
    module_fingerprints: Mapping[str, Any],
) -> None:
    """Persist compact M3 membership plus the upstream fingerprint chain."""
    membership_columns = [
        column
        for column in (
            "transcript_id",
            "cell_id",
            METASPOT_LOCAL_COL,
            METASPOT_COL,
        )
        if column in point_df.columns
    ]
    required = {"cell_id", METASPOT_LOCAL_COL, METASPOT_COL}
    if not required.issubset(membership_columns):
        raise KeyError(
            "Module 3 membership checkpoint requires cell_id, "
            f"{METASPOT_LOCAL_COL}, and {METASPOT_COL}"
        )
    path = os.path.join(
        _intermediate_dir(out_dir),
        "module3_metaspot_membership.parquet",
    )
    _atomic_dataframe_write(
        point_df.loc[:, membership_columns],
        path,
        file_format="parquet",
    )
    atomic_write_json(
        _module3_membership_manifest_path(out_dir),
        {
            "schema_version": 1,
            "rows": int(len(point_df)),
            "n_cells": int(point_df["cell_id"].nunique(dropna=False)),
            "n_metaspots": int(point_df[METASPOT_COL].nunique(dropna=False)),
            "columns": membership_columns,
            "module_fingerprints": {
                key: str(module_fingerprints.get(key, ""))
                for key in ("m1", "m2", "m3")
            },
        },
    )


def _load_module3_membership_checkpoint(
    point_df: pd.DataFrame,
    out_dir: str,
    *,
    module_fingerprints: Mapping[str, Any],
) -> Optional[pd.DataFrame]:
    """Restore M3 labels after a row-group-wise owner-cell alignment audit."""
    import json

    parquet_path = os.path.join(
        _intermediate_dir(out_dir),
        "module3_metaspot_membership.parquet",
    )
    manifest_path = _module3_membership_manifest_path(out_dir)
    if not os.path.isfile(parquet_path) or not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_fingerprints = {
            key: str(module_fingerprints.get(key, ""))
            for key in ("m1", "m2", "m3")
        }
        if manifest.get("module_fingerprints") != expected_fingerprints:
            _logging.getLogger("cellscope").warning(
                "Rejecting compact Module 3 membership: upstream module "
                "fingerprints differ"
            )
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(parquet_path)
        row_count = int(parquet.metadata.num_rows)
        if row_count != len(point_df) or int(manifest.get("rows", -1)) != row_count:
            _logging.getLogger("cellscope").warning(
                "Rejecting compact Module 3 membership: rows=%d input_rows=%d",
                row_count,
                len(point_df),
            )
            return None
        available = set(parquet.schema.names)
        required = {"cell_id", METASPOT_LOCAL_COL, METASPOT_COL}
        if not required.issubset(available):
            return None

        local_labels = np.empty(row_count, dtype=np.int32)
        metaspot_chunks = []
        offset = 0
        owner_cells = point_df["cell_id"]
        logger = _logging.getLogger("cellscope")
        logger.info(
            "Validating compact Module 3 membership | rows=%d | row_groups=%d",
            row_count,
            int(parquet.metadata.num_row_groups),
        )
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group,
                columns=["cell_id", METASPOT_LOCAL_COL, METASPOT_COL],
            )
            chunk_rows = int(table.num_rows)
            stop = offset + chunk_rows
            cached_cells = table["cell_id"].combine_chunks().to_numpy(
                zero_copy_only=False
            )
            current_cells = owner_cells.iloc[offset:stop].astype(str).to_numpy()
            if not np.array_equal(
                np.asarray(cached_cells, dtype=str),
                np.asarray(current_cells, dtype=str),
            ):
                logger.warning(
                    "Rejecting compact Module 3 membership: owner-cell order "
                    "differs in row group %d",
                    row_group,
                )
                return None
            local_labels[offset:stop] = (
                table[METASPOT_LOCAL_COL]
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
                .astype(np.int32, copy=False)
            )
            metaspot_chunks.append(table[METASPOT_COL].combine_chunks())
            offset = stop
        if offset != row_count:
            return None

        restored = point_df.copy(deep=False)
        restored[METASPOT_LOCAL_COL] = local_labels
        chunked_metaspots = pa.chunked_array(metaspot_chunks)
        try:
            restored[METASPOT_COL] = pd.Series(
                pd.arrays.ArrowStringArray(chunked_metaspots),
                index=restored.index,
            )
        except (AttributeError, TypeError):
            restored[METASPOT_COL] = chunked_metaspots.to_pandas()
        if int(restored[METASPOT_COL].nunique(dropna=False)) != int(
            manifest.get("n_metaspots", -1)
        ):
            _logging.getLogger("cellscope").warning(
                "Rejecting compact Module 3 membership: MetaSpot count differs"
            )
            return None
        logger.info(
            "Compact Module 3 membership validated and restored | "
            "MetaSpots=%d",
            int(manifest["n_metaspots"]),
        )
        return restored
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not load compact Module 3 membership: %s",
            exc,
        )
        return None


def _load_late_point_resume_checkpoint(
    out_dir: str,
    spatial_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Recover the aligned transcript-to-MetaSpot table in lite I/O mode.

    ``module5_final_data`` is deliberately persisted for resume even when the
    much larger Module 1--3 tables are omitted. It still contains every
    upstream geometry and MetaSpot membership column. Reusing it is safe only
    when its row count matches the current preprocessed input and its MetaSpot
    population is strictly aligned with the Module 4 AnnData checkpoint.
    """
    base = _intermediate_dir(out_dir)
    point_df = _load_dataframe_checkpoint(base, "module5_final_data")
    if point_df is None:
        return None
    required = {
        "cell_id",
        "feature_name",
        "x_location",
        "y_location",
        METASPOT_LOCAL_COL,
        METASPOT_COL,
    }
    missing = sorted(required.difference(point_df.columns))
    if missing or len(point_df) != len(spatial_df):
        _logging.getLogger("cellscope").warning(
            "Rejecting late point-table resume checkpoint: missing=%s | "
            "cached_rows=%d | input_rows=%d",
            missing,
            len(point_df),
            len(spatial_df),
        )
        return None
    module4 = _load_adata_checkpoint(
        os.path.join(base, "module4_adata1.h5ad")
    )
    if module4 is None:
        return None
    aligned, reason = _validate_adata1_point_df_alignment(module4, point_df)
    if not aligned:
        _logging.getLogger("cellscope").warning(
            "Rejecting late point-table resume checkpoint: %s",
            reason,
        )
        return None
    return point_df


def _load_adata_checkpoint(path: str) -> Optional[ad.AnnData]:
    if not os.path.exists(path):
        return None
    try:
        return _migrate_legacy_metaspot_obs(ad.read_h5ad(path))
    except Exception:
        return None


def _resume_build_outputs(out_dir: str) -> Optional[Tuple[ad.AnnData, Optional[ad.AnnData], Optional[pd.DataFrame]]]:
    base = _intermediate_dir(out_dir)
    a1_path = os.path.join(base, "module5_adata1.h5ad")
    adata1 = _load_adata_checkpoint(a1_path)
    if adata1 is None:
        adata1 = _load_adata_checkpoint(os.path.join(out_dir, "adata1.h5ad"))
    if adata1 is None:
        return None
    adata2 = _load_adata_checkpoint(os.path.join(base, "module6_adata2.h5ad"))
    if adata2 is None:
        adata2 = _load_adata_checkpoint(os.path.join(out_dir, "adata2.h5ad"))
    try:
        resume_cfg = load_params_yaml() or {}
        resume_graph_unit = _graph_unit_from_config(resume_cfg)
    except Exception:
        resume_graph_unit = "metadomain"
    if resume_graph_unit == "metaspot":
        if adata2 is None:
            return None
        dgi_meta = (
            adata1.uns.get("dgi_embeddings", {}).get(METASPOT_DGI_KEY, {})
        )
        if (
            METASPOT_DGI_KEY not in adata1.obsm
            or adata1.obsm[METASPOT_DGI_KEY].shape[0] != adata1.n_obs
            or dgi_meta.get("node_unit") != "metaspot"
            or int(dgi_meta.get("schema_version", -1))
            != METASPOT_DGI_SCHEMA_VERSION
            or dgi_meta.get("node_id_checksum")
            != _obs_names_checksum(adata1.obs_names)
            or METADOMAIN_DGI_KEY not in adata2.obsm
        ):
            return None
    final_df = _load_dataframe_checkpoint(base, "module5_final_data")
    used_final_output = False
    if final_df is None:
        # Lite-intermediate mode may intentionally omit the multi-million-row
        # build table even though the final Parquet/CSV contains the same
        # upstream membership columns. Reuse that final table for reannotation
        # only after strict MetaSpot ownership/count alignment checks.
        final_df = _load_dataframe_checkpoint(out_dir, "final_data")
        used_final_output = final_df is not None
    if final_df is None:
        return adata1, adata2, None

    if (
        {METASPOT_COL, "meta-domain"}.issubset(final_df.columns)
        and "meta-domain" in adata1.obs.columns
    ):
        mapping = (
            final_df[[METASPOT_COL, "meta-domain"]]
            .assign(
                **{
                    METASPOT_COL: lambda frame: frame[METASPOT_COL].astype(str),
                    "meta-domain": lambda frame: frame["meta-domain"].astype(str),
                }
            )
            .drop_duplicates()
        )
        if mapping[METASPOT_COL].duplicated().any():
            return None
        observed_mapping = mapping.set_index(METASPOT_COL)["meta-domain"]
        expected_index = pd.Index(adata1.obs_names.astype(str))
        expected_mapping = pd.Series(
            adata1.obs["meta-domain"].astype(str).to_numpy(),
            index=expected_index,
        )
        if not observed_mapping.reindex(expected_index).equals(expected_mapping):
            return None

    if used_final_output:
        if not {"metaspot", "cell_id"}.issubset(final_df.columns):
            return adata1, adata2, None
        expected_ids = pd.Index(adata1.obs_names.astype(str))
        observed_ids = pd.Index(
            pd.unique(final_df["metaspot"].astype(str))
        )
        if (
            len(expected_ids.difference(observed_ids))
            or len(observed_ids.difference(expected_ids))
        ):
            return adata1, adata2, None
        ownership_frame = pd.DataFrame(
            {
                "metaspot": final_df["metaspot"].astype(str),
                "cell_id": final_df["cell_id"].astype(str),
            }
        )
        owner_groups = ownership_frame.groupby(
            "metaspot",
            observed=True,
        )["cell_id"]
        if owner_groups.nunique().ne(1).any():
            return adata1, adata2, None
        owners = owner_groups.first()
        expected_owners = pd.Series(
            adata1.obs["cell_id"].astype(str).to_numpy(),
            index=expected_ids,
        )
        if not owners.reindex(expected_ids).equals(expected_owners):
            return adata1, adata2, None
        if "log_n" in adata1.obs.columns:
            transcript_counts = (
                final_df["metaspot"].astype(str).value_counts()
            ).reindex(expected_ids)
            expected_log_n = pd.to_numeric(
                adata1.obs["log_n"],
                errors="coerce",
            ).to_numpy(dtype=float)
            observed_log_n = np.log1p(
                transcript_counts.to_numpy(dtype=float)
            )
            if not np.allclose(
                expected_log_n,
                observed_log_n,
                rtol=1e-5,
                atol=1e-5,
                equal_nan=False,
            ):
                return adata1, adata2, None
        if (
            adata2 is not None
            and "meta-domain" in final_df.columns
        ):
            expected_domains = pd.Index(adata2.obs_names.astype(str))
            observed_domains = pd.Index(
                pd.unique(final_df["meta-domain"].astype(str))
            )
            if (
                len(expected_domains.difference(observed_domains))
                or len(observed_domains.difference(expected_domains))
            ):
                return adata1, adata2, None
        _logging.getLogger("cellscope").info(
            "Resume: using aligned final_data as the build-table checkpoint "
            "(lite-intermediate mode)"
        )
    return adata1, adata2, final_df


def _resume_annotation_outputs(
    out_dir: str,
) -> Optional[
    Tuple[
        Optional[ad.AnnData],
        Optional[ad.AnnData],
        Optional[pd.DataFrame],
    ]
]:
    base = _intermediate_dir(out_dir)
    final_df = _load_dataframe_checkpoint(base, "annotated_final_csv")
    if final_df is None:
        return None
    ann_adata1 = _load_adata_checkpoint(
        os.path.join(base, "annotated_adata1.h5ad")
    )
    ann_adata2 = _load_adata_checkpoint(os.path.join(base, "annotated_adata2.h5ad"))
    return ann_adata1, ann_adata2, final_df


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
    common_cells = (
        set(spatial_df["cell_id"].dropna())
        & set(cell_df["cell_id"].dropna())
        & set(nucleus_df["cell_id"].dropna())
    )

    if use_input_nucleus_flags:
        if "overlaps_nucleus" not in spatial_df.columns:
            raise ValueError(
                "spatial input is missing overlaps_nucleus while "
                "module1.use_input_nucleus_flags=true"
            )
        unique_counts = spatial_df.groupby("cell_id", observed=True)[
            "overlaps_nucleus"
        ].nunique(dropna=True)
        common_cells &= set(unique_counts[unique_counts == 2].index)

    spatial_df = spatial_df[spatial_df["cell_id"].isin(common_cells)].copy()
    cell_df = cell_df[cell_df["cell_id"].isin(common_cells)].copy()
    nucleus_df = nucleus_df[nucleus_df["cell_id"].isin(common_cells)].copy()

    return spatial_df, cell_df, nucleus_df


def _stretch_signed_geometry(values: np.ndarray, cell_size: int) -> np.ndarray:
    """Robustly scale signed geometry while keeping the nuclear boundary at zero."""
    x = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return x

    if cell_size >= 16:
        negative = finite & (x < 0)
        positive = finite & (x > 0)
        if negative.any():
            neg_scale = float(np.nanpercentile(np.abs(x[negative]), 95.0))
            if np.isfinite(neg_scale) and neg_scale > 1e-6:
                x[negative] /= neg_scale
        if positive.any():
            pos_scale = float(np.nanpercentile(x[positive], 95.0))
            if np.isfinite(pos_scale) and pos_scale > 1e-6:
                x[positive] /= pos_scale

    x = np.clip(x, -2, 2)
    adaptive_band = 0.05 * (1.0 + 0.5 * np.exp(-cell_size / 1000))
    x[np.abs(x) < adaptive_band] = 0.0
    return x.astype(np.float32, copy=False)


def _point_to_ring_distance(
    points: np.ndarray,
    ring: np.ndarray,
    chunk_size: int = 20000,
) -> np.ndarray:
    """Exact distance from each point to the edges of a closed polygon ring."""
    pts = np.asarray(points, dtype=np.float64)
    start = np.asarray(ring, dtype=np.float64)
    end = np.roll(start, -1, axis=0)
    edge = end - start
    edge_sq = np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-24)
    out = np.empty(pts.shape[0], dtype=np.float64)
    for lo in range(0, pts.shape[0], chunk_size):
        rel = pts[lo:lo + chunk_size, None, :] - start[None, :, :]
        t = np.clip(np.einsum("nmj,mj->nm", rel, edge) / edge_sq, 0.0, 1.0)
        gap = rel - t[:, :, None] * edge[None, :, :]
        out[lo:lo + chunk_size] = np.sqrt(
            np.einsum("nmj,nmj->nm", gap, gap).min(axis=1)
        )
    return out


def build_point_geometry_v2(
    transcripts_df: pd.DataFrame,
    nucleus_boundaries_df: pd.DataFrame,
    cell_boundaries_df: pd.DataFrame,
    sampled_cells: List,
    include_directional: bool = True,
    use_input_nucleus_flags: bool = True,
    winsor_perc=(1, 99),
    show_internal_progress: bool = False,
    boundary_distance: str = "vertex",
) -> pd.DataFrame:
    DTYPE = np.float32
    boundary_distance = str(boundary_distance or "vertex").strip().lower()
    if boundary_distance not in {"vertex", "edge"}:
        raise ValueError("module1.boundary_distance must be 'vertex' or 'edge'")
    use_edge_distance = boundary_distance == "edge"

    def _group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _winsorize_inplace(df: pd.DataFrame, cols, perc=(1, 99)) -> None:
        lo_p, hi_p = perc
        for c in cols:
            x = df[c].to_numpy()
            lo, hi = np.nanpercentile(x, [lo_p, hi_p])
            df[c] = np.clip(x, lo, hi).astype(DTYPE, copy=False)

    def _build_kdtrees(boundary_df: pd.DataFrame, cells, desc="KDTree") -> dict:
        kd = {}
        sub = boundary_df[boundary_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(it, desc=f'Building {desc}', total=it.ngroups, show=show_internal_progress)
        for cid, g in iterator:
            coords = g[['vertex_x', 'vertex_y']].to_numpy()
            kd[cid] = KDTree(coords) if len(coords) else None
        return kd

    def _compute_min_dist(df_points: pd.DataFrame, kdtrees: dict, label: str) -> pd.Series:
        parts = []
        it = df_points.groupby('cell_id', sort=False)
        iterator = _progress_iter(it, desc='Compute '+label, total=it.ngroups, show=show_internal_progress)
        for cid, g in iterator:
            kt = kdtrees.get(cid, None)
            if kt is None:
                parts.append(pd.Series(np.nan, index=g.index))
            else:
                q = g[['x_location', 'y_location']].to_numpy()
                d, _ = kt.query(q, k=1, return_distance=True)
                parts.append(pd.Series(d[:, 0].astype(DTYPE), index=g.index))
        return pd.concat(parts)

    def _build_ordered_polygons(boundary_df: pd.DataFrame, cells, desc: str) -> dict:
        """Build valid ordered polygons from boundary-point tables.

        Some spatial formats store boundary pixels/vertices without contour
        ordering. ``matplotlib.path.Path`` assumes ordered polygon vertices;
        feeding an unordered point cloud can collapse the inferred nuclear
        area and severely reduce nuclear recall. We preserve apparently
        ordered contours and reconstruct clearly unordered contours with a
        convex hull. Nuclei are typically close to convex, making this a
        conservative and stable fallback for cross-platform input.
        """
        from scipy.spatial import ConvexHull
        from scipy.spatial import QhullError

        polygons = {}
        sub = boundary_df[boundary_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(
            it,
            desc=desc,
            total=it.ngroups,
            show=show_internal_progress,
        )

        for cid, g in iterator:
            pts = (
                g[['vertex_x', 'vertex_y']]
                .apply(pd.to_numeric, errors='coerce')
                .dropna()
                .to_numpy(dtype=np.float64)
            )
            # ``np.unique(..., axis=0)`` sorts rows, which destroys contour
            # order. Keep the first occurrence of each vertex instead.
            if len(pts):
                _, first_idx = np.unique(pts, axis=0, return_index=True)
                pts = pts[np.sort(first_idx)]

            if len(pts) < 3:
                polygons[cid] = None
                continue

            try:
                tree = KDTree(pts)
                nn_dist, _ = tree.query(pts, k=2, return_distance=True)
                local_scale = float(np.nanmedian(nn_dist[:, 1]))
            except Exception:
                local_scale = np.nan

            closed_pts = np.vstack([pts, pts[:1]])
            path_steps = np.sqrt(
                np.sum(np.diff(closed_pts, axis=0) ** 2, axis=1)
            )
            typical_step = float(np.nanmedian(path_steps))
            upper_step = float(np.nanpercentile(path_steps, 95))

            ordered = (
                np.isfinite(local_scale)
                and local_scale > 0
                and typical_step <= 3.0 * local_scale
                and upper_step <= 10.0 * local_scale
            )

            polygon_pts = pts

            # A short vertex list can pass the step-size heuristic while still
            # being self-intersecting (for example, bow-tie ordering). Treat
            # only a valid, non-zero-area Shapely polygon as an ordered contour.
            if ordered:
                try:
                    from shapely.geometry import Polygon as _Polygon

                    candidate = _Polygon(pts)
                    ordered = bool(
                        candidate.is_valid
                        and not candidate.is_empty
                        and float(candidate.area) > 0.0
                    )
                except Exception:
                    ordered = False

            if not ordered:
                try:
                    hull = ConvexHull(pts)
                    polygon_pts = pts[hull.vertices]
                except QhullError:
                    center = np.nanmedian(pts, axis=0)
                    angle = np.arctan2(
                        pts[:, 1] - center[1],
                        pts[:, 0] - center[0],
                    )
                    polygon_pts = pts[np.argsort(angle)]

            polygons[cid] = polygon_pts if len(polygon_pts) >= 3 else None

        return polygons

    def _polygon_paths(polygons: dict) -> dict:
        return {
            # With ``closed=True`` Matplotlib treats the last vertex as a
            # CLOSEPOLY placeholder. Repeat the first point explicitly so
            # the final real vertex is not discarded.
            cid: (
                Path(np.vstack([poly, poly[:1]]), closed=True)
                if poly is not None
                else None
            )
            for cid, poly in polygons.items()
        }

    def _boundary_distance(q: np.ndarray, kt, polygon) -> np.ndarray:
        if use_edge_distance and polygon is not None:
            return _point_to_ring_distance(q, polygon).astype(DTYPE)
        d, _ = kt.query(q, k=1, return_distance=True)
        return d[:, 0].astype(DTYPE)

    # Avoid copying the full transcript table. We keep the original rows and
    # build geometry columns in chunks so Module 1 can run without doubling RAM.
    df = transcripts_df[transcripts_df['cell_id'].isin(sampled_cells)]
    cell_kd = _build_kdtrees(cell_boundaries_df, sampled_cells, desc="Cell KDTree")
    nuc_kd = _build_kdtrees(nucleus_boundaries_df, sampled_cells, desc="Nucleus KDTree")
    use_input_nucleus_flags = bool(
        use_input_nucleus_flags
        and "overlaps_nucleus" in df.columns
    )
    need_nuc_polygons = use_edge_distance or not use_input_nucleus_flags
    nuc_polygons = (
        _build_ordered_polygons(
            nucleus_boundaries_df, sampled_cells, 'Build nucleus polygons'
        )
        if need_nuc_polygons
        else {}
    )
    cell_polygons = (
        _build_ordered_polygons(
            cell_boundaries_df, sampled_cells, 'Build cell polygons'
        )
        if use_edge_distance
        else {}
    )
    nuc_paths = {} if use_input_nucleus_flags else _polygon_paths(nuc_polygons)

    # Process grouped rows in chunks to avoid materializing several full-size
    # temporary Series at once.
    d_cell_parts = []
    d_nuc_parts = []
    flags = []
    group_iter = _group_chunks(df, chunk_size=128)
    total_groups = int(df['cell_id'].nunique(dropna=False)) if len(df) else 0
    iterator = _progress_iter(group_iter, desc='Point geometry chunks', total=max(1, (total_groups + 127) // 128), show=show_internal_progress)
    for chunk in iterator:
        for cid, g in chunk:
            kt = cell_kd.get(cid, None)
            if kt is None:
                d_cell_parts.append(pd.Series(np.nan, index=g.index))
            else:
                q = g[['x_location', 'y_location']].to_numpy()
                d_cell_parts.append(pd.Series(
                    _boundary_distance(q, kt, cell_polygons.get(cid)),
                    index=g.index,
                ))

            kt = nuc_kd.get(cid, None)
            if kt is None:
                d_nuc_parts.append(pd.Series(np.nan, index=g.index))
            else:
                q = g[['x_location', 'y_location']].to_numpy()
                d_nuc_parts.append(pd.Series(
                    _boundary_distance(q, kt, nuc_polygons.get(cid)),
                    index=g.index,
                ))

            if use_input_nucleus_flags:
                input_flags = (
                    pd.to_numeric(g["overlaps_nucleus"], errors="coerce")
                    .fillna(0)
                    .ne(0)
                )
                flags.append(pd.Series(input_flags.to_numpy(), index=g.index))
            else:
                p = nuc_paths.get(cid, None)
                if p is None:
                    flags.append(pd.Series(False, index=g.index))
                else:
                    pts = g[['x_location', 'y_location']].to_numpy()
                    flags.append(pd.Series(p.contains_points(pts), index=g.index))

    df['d_cell'] = pd.concat(d_cell_parts)
    df['d_nuc'] = pd.concat(d_nuc_parts)
    df['in_nucleus'] = pd.concat(flags)

    d_cell = df['d_cell'].to_numpy(dtype=DTYPE, copy=False)
    d_nuc = df['d_nuc'].to_numpy(dtype=DTYPE, copy=False)
    denom = (d_cell + d_nuc).astype(DTYPE, copy=False)
    denom[denom <= 0] = np.nan

    inside = df['in_nucleus'].to_numpy()
    df['r_signed'] = np.where(inside, -d_cell/denom, d_nuc/denom).astype(DTYPE, copy=False)
    df['d_cell_norm'] = (d_cell/denom).astype(DTYPE, copy=False)
    df['d_nuc_norm'] = (d_nuc/denom).astype(DTYPE, copy=False)

    if include_directional:
        nuc_cent = (nucleus_boundaries_df[nucleus_boundaries_df['cell_id'].isin(sampled_cells)]
                    .groupby('cell_id', sort=False)
                    .agg(nuc_x=('vertex_x', 'mean'), nuc_y=('vertex_y', 'mean')))
        df = df.join(nuc_cent, on='cell_id')
        nan_c = df['nuc_x'].isna() | df['nuc_y'].isna()
        if nan_c.any():
            fb = (df.loc[nan_c]
                  .groupby('cell_id')[['x_location', 'y_location']]
                  .mean()
                  .rename(columns={'x_location': 'nuc_x', 'y_location': 'nuc_y'}))
            df.update(fb, overwrite=False)
        dx = (df['x_location'] - df['nuc_x']).to_numpy(dtype=DTYPE, copy=False)
        dy = (df['y_location'] - df['nuc_y']).to_numpy(dtype=DTYPE, copy=False)
        norm = np.sqrt(dx**2 + dy**2)
        norm[norm <= 0] = np.nan
        df['cos_theta'] = (dx / norm).astype(DTYPE, copy=False)
        df['sin_theta'] = (dy / norm).astype(DTYPE, copy=False)
        st = df['sin_theta'].to_numpy(dtype=DTYPE, copy=False)
        ct = df['cos_theta'].to_numpy(dtype=DTYPE, copy=False)
        df['sin2_theta'] = (2.0 * st * ct).astype(DTYPE, copy=False)
        df['cos2_theta'] = (ct*ct - st*st).astype(DTYPE, copy=False)

    # Nuclear radius estimation and per-cell stretching
    nuc_cent_xy = (nucleus_boundaries_df.groupby('cell_id', sort=False)
                   .agg(_nx=('vertex_x', 'mean'), _ny=('vertex_y', 'mean')))
    nuc_bd = (nucleus_boundaries_df
              .join(nuc_cent_xy, on='cell_id')
              .assign(_r=lambda t: np.sqrt((t['vertex_x']-t['_nx'])**2 + (t['vertex_y']-t['_ny'])**2)))
    rad_df = nuc_bd.groupby('cell_id', sort=False)['_r'].median().rename('_nuc_r')
    df = df.join(rad_df, on='cell_id')

    _nuc_r = df['_nuc_r'].to_numpy(dtype=DTYPE, copy=False)
    _nuc_r = np.where((~np.isfinite(_nuc_r)) | (_nuc_r <= 0), np.nan, _nuc_r)
    rs_norm = df['r_signed'].to_numpy(dtype=DTYPE, copy=False) / np.nan_to_num(_nuc_r, nan=1.0)

    df['_rs_tmp'] = rs_norm.astype(DTYPE, copy=False)
    cell_sizes = df.groupby('cell_id').size()
    df['_cell_size'] = df['cell_id'].map(cell_sizes)
    r_parts = []
    group_iter = _group_chunks(df, chunk_size=128)
    iterator = _progress_iter(group_iter, desc='Stretch r_signed chunks', total=max(1, (total_groups + 127) // 128), show=show_internal_progress)
    for chunk in iterator:
        for cid, g in chunk:
            r_parts.append(pd.Series(_stretch_signed_geometry(g['_rs_tmp'].values, len(g)), index=g.index))
    df['r_signed'] = pd.concat(r_parts).astype(DTYPE)
    df.drop(columns=['_rs_tmp', '_nuc_r', '_cell_size'], errors='ignore', inplace=True)

    _winsorize_inplace(df, ['d_cell', 'd_nuc', 'r_signed', 'd_cell_norm', 'd_nuc_norm'], winsor_perc)
    return df


def build_composition_embeddings(point_df: pd.DataFrame, transcripts_df: pd.DataFrame, sampled_cell_ids: List, show_internal_progress: bool = False) -> Tuple[pd.DataFrame, List[str], Optional[Any]]:
    RANDOM_STATE = 42
    # Build vocabulary
    # Keep only the columns Module 2 actually needs; carrying the full point table
    # through this stage is a major memory multiplier.
    # Avoid huge memory spikes when sampled_cell_ids is a full row-level list.
    use_full_source = False
    try:
        use_full_source = (
            sampled_cell_ids is None
            or len(sampled_cell_ids) >= transcripts_df['cell_id'].nunique(dropna=False)
        )
    except Exception:
        use_full_source = False
    if use_full_source:
        sub_tr_all = transcripts_df
    else:
        sampled_ids = pd.unique(np.asarray(sampled_cell_ids))
        sub_tr_all = transcripts_df[transcripts_df['cell_id'].isin(sampled_ids)]
    gene_counts = sub_tr_all['feature_name'].astype(str).value_counts()
    # defaults; will be overridden by config when available
    TOP_G = 384
    N_COMP = 24
    K_NEI = 32
    MAX_ROWS_PER_CELL_FOR_SVD = 600
    MAX_SVD_ROWS = 120_000
    BOW_WEIGHT_MODE = 'gaussian'  # gaussian | uniform
    USE_TFIDF = True
    TF_NORM = True
    IDF_MODE = 'log1p'  # 'log1p' | 'log' | 'none'
    STREAM_LOCAL = True
    PARALLEL_CHUNK_CELLS = 128
    try:
        _cfg = load_params_yaml()
        _m2 = _cfg.get('module2', {}) if _cfg else {}
        TOP_G = int(_m2.get('top_g', TOP_G))
        N_COMP = int(_m2.get('svd_n_components', N_COMP))
        K_NEI = int(_m2.get('bow_knn_k', K_NEI))
        MAX_ROWS_PER_CELL_FOR_SVD = int(_m2.get('max_rows_per_cell_for_svd', MAX_ROWS_PER_CELL_FOR_SVD))
        MAX_SVD_ROWS = int(_m2.get('max_svd_rows', MAX_SVD_ROWS))
        BOW_WEIGHT_MODE = str(_m2.get('bow_weight_mode', BOW_WEIGHT_MODE)).lower()
        USE_TFIDF = bool(_m2.get('use_tfidf', USE_TFIDF))
        TF_NORM = bool(_m2.get('tf_norm', TF_NORM))
        IDF_MODE = str(_m2.get('idf_mode', IDF_MODE)).lower()
        STREAM_LOCAL = bool(_m2.get('stream_local_embeddings', STREAM_LOCAL))
        PARALLEL_CHUNK_CELLS = max(1, int(_m2.get('parallel_chunk_cells', PARALLEL_CHUNK_CELLS)))
    except Exception:
        pass
    vocab_genes = gene_counts.head(TOP_G).index.astype(str).tolist()
    gene2idx = {g: i for i, g in enumerate(vocab_genes)}

    # For real runs, the old per-transcript KNN + BoW path still produces large
    # temporary matrices. A cell-level summary embedding is much cheaper and is
    # sufficient for downstream cell/subcell clustering because the geometry
    # features already stay transcript-level.
    force_cell_summary = False
    try:
        force_cell_summary = os.environ.get('CELLSCOPE_CELL_SUMMARY_ONLY', '').lower() in ('1', 'true', 'yes')
    except Exception:
        force_cell_summary = False

    if force_cell_summary:
        cell_ids = []
        row_idx = []
        col_idx = []
        values = []
        local_density = []
        df_vec = np.zeros(len(vocab_genes), dtype=np.float32)

        group_iter = _progress_iter(
            transcripts_df.groupby('cell_id', sort=False),
            desc='Cell composition summary',
            total=int(transcripts_df['cell_id'].nunique(dropna=False)) if len(transcripts_df) else 0,
            show=show_internal_progress,
        )
        for cid, dfc in group_iter:
            feat = dfc['feature_name'].astype(str).map(gene2idx).fillna(-1).to_numpy(dtype=np.int32, copy=False)
            feat = feat[feat >= 0]
            if feat.size == 0:
                continue
            counts = np.bincount(feat, minlength=len(vocab_genes)).astype(np.float32, copy=False)
            nz = np.flatnonzero(counts)
            r = len(cell_ids)
            cell_ids.append(cid)
            row_idx.extend([r] * len(nz))
            col_idx.extend(nz.tolist())
            values.extend(counts[nz].tolist())
            df_vec[nz] += 1.0
            local_density.append(float(np.log1p(len(dfc))))

        if len(cell_ids) == 0:
            raise RuntimeError('Cell summary composition samples empty.')

        cell_bow = sp.csr_matrix(
            (np.asarray(values, dtype=np.float32), (np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64))),
            shape=(len(cell_ids), len(vocab_genes)),
            dtype=np.float32,
        )

        N_rows = float(cell_bow.shape[0])
        if USE_TFIDF and N_rows > 0:
            if IDF_MODE == 'none':
                idf_vec = np.ones_like(df_vec, dtype=np.float32)
            elif IDF_MODE == 'log':
                idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
            else:
                idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
            cell_bow = cell_bow.multiply(idf_vec[np.newaxis, :]).tocsr()

        gc.collect()
        if os.environ.get('CELLSCOPE_ULTRALOW', '').lower() in ('1', 'true', 'yes') or cell_bow.shape[0] > 50000:
            from sklearn.random_projection import SparseRandomProjection
            emb_model = SparseRandomProjection(n_components=N_COMP, random_state=RANDOM_STATE)
            emb_model.fit(cell_bow)
            transformer = emb_model
        else:
            svd_local = TruncatedSVD(n_components=N_COMP, random_state=RANDOM_STATE)
            svd_local.fit(cell_bow)
            transformer = svd_local

        # Transform in batches to avoid materializing the full dense cell embedding
        # matrix at once. The cell-level output stays small enough to concatenate.
        batch_size = max(1024, min(8192, int(np.ceil(cell_bow.shape[0] / 32))))
        cell_emb_parts = []
        for start in range(0, cell_bow.shape[0], batch_size):
            end = min(start + batch_size, cell_bow.shape[0])
            comp_part = transformer.transform(cell_bow[start:end])
            if sp.issparse(comp_part):
                comp_part = comp_part.toarray()
            comp_part = np.asarray(comp_part, dtype=np.float32)
            if comp_part.ndim == 1:
                comp_part = comp_part.reshape(-1, 1)
            if comp_part.shape[1] == 0:
                comp_part = np.zeros((comp_part.shape[0], 1), dtype=np.float32)
            comp_cols = [f'comp_emb{i}' for i in range(comp_part.shape[1])]
            part_index = pd.Index(cell_ids[start:end], name='cell_id')
            part_df = pd.DataFrame(comp_part, index=part_index, columns=comp_cols)
            part_df['local_density'] = np.asarray(local_density[start:end], dtype=np.float32)
            cell_emb_parts.append(part_df)

        cell_emb = pd.concat(cell_emb_parts, axis=0) if len(cell_emb_parts) > 1 else cell_emb_parts[0]
        comp_cols = [c for c in cell_emb.columns if str(c).startswith('comp_emb')]

        geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
            ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
        )
        gmm_cols = geom_cols + comp_cols + ['local_density']
        del cell_bow, df_vec, values, row_idx, col_idx, local_density
        gc.collect()
        return point_df, gmm_cols, cell_emb

    def _build_knn_per_cell(dfc: pd.DataFrame):
        pts = dfc[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
        if len(pts) == 0:
            return None, None
        ind, dist = _kdtree_query(pts, k=min(K_NEI, len(dfc)))
        return ind, dist

    def _nei_bow_for_indices(
        dfc: pd.DataFrame,
        ind_mat: np.ndarray,
        dist_mat: np.ndarray = None,
        weight_mode: str = 'gaussian',
    ) -> sp.csr_matrix:
        """Vectorized neighborhood BoW with optional distance weights (sparse CSR).
        weight_mode: 'gaussian' | 'inv' | 'uniform'
        """
        G = len(vocab_genes)
        n = len(dfc)
        if n == 0 or G == 0:
            return sp.csr_matrix((n, G), dtype=np.float32)

        gene_idx = (
            dfc['feature_name']
               .astype(str)
               .map(gene2idx).fillna(-1)
               .astype(np.int64, copy=False)
               .to_numpy(copy=False)
        )

        ind = np.asarray(ind_mat)
        if ind.ndim == 1:
            ind = ind[None, :]
        ind = ind.astype(np.int64, copy=False)
        k = ind.shape[1]

        if dist_mat is not None:
            D = np.asarray(dist_mat, dtype=np.float32)
            if D.ndim == 1:
                D = D[None, :]
            D = np.clip(D, 0.0, np.inf)
        else:
            D = None

        # Build weight matrix W aligned with ind's shape
        if D is not None:
            if weight_mode == 'gaussian':
                rk = D[:, min(k - 1, D.shape[1] - 1)].astype(np.float32) + 1e-6
                scale = rk[:, None]
                W = np.exp(- (D / scale) ** 2).astype(np.float32, copy=False)
            elif weight_mode == 'inv':
                W = (1.0 / (D + 1e-3)).astype(np.float32, copy=False)
            else:
                W = np.ones_like(D, dtype=np.float32)
        else:
            W = np.ones_like(ind, dtype=np.float32)

        # Flatten and filter invalid neighbor rows/genes
        row_idx = np.repeat(np.arange(n, dtype=np.int64), k)
        nei_flat = ind.ravel()
        w_flat = W.ravel()

        valid = (nei_flat >= 0) & (nei_flat < n)
        if not valid.any():
            return sp.csr_matrix((n, G), dtype=np.float32)
        nei_flat = nei_flat[valid]
        row_idx = row_idx[valid]
        w_flat = w_flat[valid]

        gi_flat = gene_idx[nei_flat]
        gene_valid = (gi_flat >= 0) & (gi_flat < G)
        if not gene_valid.any():
            return sp.csr_matrix((n, G), dtype=np.float32)

        gi_flat = gi_flat[gene_valid]
        row_idx = row_idx[gene_valid]
        w_flat = w_flat[gene_valid]

        # Build sparse BoW via COO then convert to CSR
        bow = sp.coo_matrix((w_flat.astype(np.float32, copy=False), (row_idx, gi_flat)), shape=(n, G))
        return bow.tocsr()
    # Sample to build SVD training set
    rng = np.random.default_rng(RANDOM_STATE)
    from joblib import Parallel, delayed
    # speed-oriented parallel settings (bounded without env vars)
    # Keep module 2 conservative on memory: use at most 2 workers for large runs.
    N_JOBS = max(1, min(2, _preferred_n_jobs()))
    _backend = _joblib_backend()
    unique_cells = pd.unique(sub_tr_all['cell_id'])
    sample_cell_n = len(unique_cells)
    try:
        _cfg = load_params_yaml() or {}
        _m2 = _cfg.get('module2', {}) if _cfg else {}
        svd_sample_cells = _m2.get('svd_sample_cells')
        if isinstance(svd_sample_cells, (int, float)) and int(svd_sample_cells) > 0:
            sample_cell_n = min(sample_cell_n, int(svd_sample_cells))
    except Exception:
        pass
    # Extra safety caps if explicit ultralow env set or very constrained workers
    try:
        if os.environ.get('CELLSCOPE_ULTRALOW', '').lower() in ('1', 'true', 'yes'):
            sample_cell_n = min(sample_cell_n, 128)
            MAX_SVD_ROWS = min(MAX_SVD_ROWS, 20000)
    except Exception:
        pass
    # Keep complete cells so the fitted basis sees real spatial neighborhoods.
    # The row budget bounds both the number of cells and retained SVD documents.
    max_cells_by_rows = max(1, MAX_SVD_ROWS // max(1, MAX_ROWS_PER_CELL_FOR_SVD))
    sample_cell_n = max(1, min(int(sample_cell_n), len(unique_cells), max_cells_by_rows))
    sample_cells = rng.choice(unique_cells, size=sample_cell_n, replace=False)

    def _iter_group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    sub_tr_sample = sub_tr_all[sub_tr_all['cell_id'].isin(sample_cells)]
    n_groups_sample = int(sub_tr_sample['cell_id'].nunique(dropna=False)) if len(sub_tr_sample) else 0

    def _svd_tf_block(args):
        cid, dfc = args
        ind, dist = _build_knn_per_cell(dfc)
        if ind is None:
            return None
        bow = _nei_bow_for_indices(dfc, ind, dist, weight_mode=BOW_WEIGHT_MODE)
        if TF_NORM:
            row_sum = np.asarray(bow.sum(1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            tf = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        else:
            tf = bow.tocsr()
        if tf.shape[0] > MAX_ROWS_PER_CELL_FOR_SVD:
            idx_loc = rng.choice(tf.shape[0], size=MAX_ROWS_PER_CELL_FOR_SVD, replace=False)
            tf = tf[idx_loc]
        return tf

    bow_samples = []
    total_rows = 0
    # accumulate document frequency per gene for IDF without storing all blocks
    df_vec = np.zeros(len(vocab_genes), dtype=np.float32)
    use_rp = bool(STREAM_LOCAL)
    sample_chunk_size = PARALLEL_CHUNK_CELLS
    n_sample_chunks = (n_groups_sample + sample_chunk_size - 1) // sample_chunk_size if n_groups_sample else 0
    iterator = _progress_iter(
        _iter_group_chunks(sub_tr_sample, sample_chunk_size),
        desc='SVD sample TF (parallel)',
        total=n_sample_chunks,
        show=show_internal_progress,
    )
    for chunk in iterator:
        out = Parallel(n_jobs=N_JOBS, backend=_backend, verbose=0)(
            delayed(_svd_tf_block)(pair) for pair in chunk
        )
        for tf in out:
            if tf is None:
                continue
            # update df counts and optionally keep block
            df_vec += np.asarray(tf.getnnz(axis=0), dtype=np.float32)
            if total_rows < MAX_SVD_ROWS and not STREAM_LOCAL:
                bow_samples.append(tf)
            total_rows += tf.shape[0]
            if total_rows >= MAX_SVD_ROWS:
                break
        if total_rows >= MAX_SVD_ROWS:
            break

    if len(bow_samples) == 0 and not STREAM_LOCAL:
        raise RuntimeError('SVD samples empty.')

    # Decide whether to use TruncatedSVD or fall back to SparseRandomProjection
    # to avoid vstack of many sparse blocks which can peak memory.
    if total_rows > MAX_SVD_ROWS or os.environ.get('CELLSCOPE_ULTRALOW', '').lower() in ('1', 'true', 'yes'):
        use_rp = True

    if not use_rp:
        B_sample = sp.vstack(bow_samples, format='csr', dtype=np.float32)
        # Optional TF-IDF reweighting for SVD (sparse)
        if USE_TFIDF and B_sample.shape[0] > 0:
            df_vec = np.asarray(B_sample.getnnz(axis=0), dtype=np.float32)
            N_rows = float(B_sample.shape[0])
            if IDF_MODE == 'none':
                idf_vec = np.ones_like(df_vec, dtype=np.float32)
            elif IDF_MODE == 'log':
                idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
            else:  # 'log1p'
                idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
            B_sample = B_sample.multiply(idf_vec[np.newaxis, :]).tocsr()
        # Fit SVD on the sampled (sparse) block. GC before fit to reduce peaks.
        gc.collect()
        svd_local = TruncatedSVD(n_components=N_COMP, random_state=RANDOM_STATE)
        svd_local.fit(B_sample)
    else:
        # Low-memory fallback: sparse random projection
        from sklearn.random_projection import SparseRandomProjection
        N_rows = float(max(1, total_rows))
        if IDF_MODE == 'none':
            idf_vec = np.ones_like(df_vec, dtype=np.float32)
        elif IDF_MODE == 'log':
            idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
        else:
            idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
        rp_components = max(1, min(N_COMP, len(vocab_genes)))
        rp = SparseRandomProjection(n_components=rp_components, random_state=RANDOM_STATE)
        rp.fit(np.zeros((1, len(vocab_genes)), dtype=np.float32))
        svd_local = rp

    if STREAM_LOCAL:
        model_idf = (
            idf_vec
            if USE_TFIDF
            else np.ones(len(vocab_genes), dtype=np.float32)
        )
        local_model = _LocalCompositionModel(
            gene2idx=gene2idx,
            transformer=svd_local,
            idf_vec=model_idf,
            n_components=rp_components if use_rp else N_COMP,
            knn_k=K_NEI,
            weight_mode=BOW_WEIGHT_MODE,
            tf_norm=TF_NORM,
            use_tfidf=USE_TFIDF,
        )
        geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
            ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
        )
        gmm_cols = geom_cols + local_model.feature_names + ['local_density']
        del bow_samples
        if 'B_sample' in locals():
            del B_sample
        gc.collect()
        return point_df, gmm_cols, local_model

    # Embed all rows (full pass)
    # Avoid copying the full table to reduce peak memory; rely on index positions.
    sub_tr = sub_tr_all
    emb_dtype = np.float32
    try:
        # For very large runs, float16 greatly reduces memory and is sufficient
        # because downstream clustering re-casts to float32 before fitting.
        if len(sub_tr) >= 5_000_000:
            emb_dtype = np.float16
    except Exception:
        pass
    n_groups_full = int(sub_tr['cell_id'].nunique(dropna=False)) if len(sub_tr) else 0

    def _emb_block(args):
        cid, dfc = args
        ind, dist = _build_knn_per_cell(dfc)
        if ind is None:
            return None
        bow = _nei_bow_for_indices(dfc, ind, dist, weight_mode=BOW_WEIGHT_MODE)
        if TF_NORM:
            row_sum = np.asarray(bow.sum(1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            tf = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        else:
            tf = bow.tocsr()
        if USE_TFIDF:
            # Use same IDF as fitted on the sample
            tfidf = tf.multiply(idf_vec[np.newaxis, :]).tocsr()
            Z = svd_local.transform(tfidf).astype(np.float32, copy=False)
        else:
            Z = svd_local.transform(tf).astype(np.float32, copy=False)
        if sp.issparse(Z):
            Z = Z.toarray()
        k_used = min(K_NEI, len(dfc))
        if dist is not None and dist.shape[1] >= k_used:
            rk = dist[:, k_used - 1].astype(np.float32) + 1e-3
            dens = (k_used) / (np.pi * rk * rk)
        else:
            dens = np.full(len(dfc), k_used, np.float32)
        # Use the dataframe index positions as row ids to avoid adding columns/copies
        rows = dfc.index.to_numpy(dtype=np.int64, copy=False)
        return rows, Z, dens

    chunk_size2 = PARALLEL_CHUNK_CELLS
    n_chunks_full = (n_groups_full + chunk_size2 - 1) // chunk_size2 if n_groups_full else 0
    iterator2 = _progress_iter(
        _iter_group_chunks(sub_tr, chunk_size2),
        desc='Per-cell KNN & BoW (parallel)',
        total=n_chunks_full,
        show=show_internal_progress,
    )
    comp_cols = None
    for chunk in iterator2:
        try:
            out = Parallel(
                n_jobs=N_JOBS,
                backend=_backend,
                verbose=0,
                pre_dispatch=f"2*{N_JOBS}",
                batch_size="auto",
                # Avoid memmap on small arrays to reduce overhead
                max_nbytes=None,
            )(
                delayed(_emb_block)(pair) for pair in chunk
            )
        except Exception as e:
            print(f"[WARN] Parallel KNN/BoW chunk failed, fallback sequential: {e}")
            out = [ _emb_block(pair) for pair in chunk ]
        for item in out:
            if item is None:
                continue
            rows, Z, dens = item
            Z = np.asarray(Z, dtype=emb_dtype)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            if Z.shape[1] == 0:
                Z = np.zeros((Z.shape[0], 1), dtype=emb_dtype)
            if comp_cols is None:
                comp_cols = [f'comp_emb{i}' for i in range(Z.shape[1])]
            elif Z.shape[1] != len(comp_cols):
                if Z.shape[1] < len(comp_cols):
                    pad = np.zeros((Z.shape[0], len(comp_cols) - Z.shape[1]), dtype=emb_dtype)
                    Z = np.concatenate([Z, pad], axis=1)
                else:
                    Z = Z[:, :len(comp_cols)]
            point_df.loc[rows, comp_cols] = Z
            point_df.loc[rows, 'local_density'] = np.asarray(dens, dtype=emb_dtype)

    if comp_cols is None:
        comp_cols = [f'comp_emb{i}' for i in range(N_COMP)]

    # Do not materialize gmm_feat_* dense columns here: this can double memory on
    # ultra-large tables. Per-cell MetaSpot segmentation consumes gmm_cols and performs
    # per-cell standardization internally.
    geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
        ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
    )
    gmm_cols = geom_cols + comp_cols + ['local_density']
    del B_sample, bow_samples
    gc.collect()
    return point_df, gmm_cols, None


def _safe_standardize(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    mu = X.mean(0, dtype=np.float64)
    sd = X.std(0, dtype=np.float64)
    good = sd > 0
    if not np.any(good):
        return np.zeros_like(X, np.float32), good
    Xg = X[:, good].astype(np.float32, copy=False)
    Xz = (Xg - mu[good].astype(np.float32)) / sd[good].astype(np.float32)
    return Xz, good


def _select_top_variance_features(X: np.ndarray, n_features: int) -> np.ndarray:
    """Select per-cell features without creating a dataset-wide dense matrix."""
    X = np.asarray(X, dtype=np.float32)
    limit = int(n_features)
    if limit <= 0 or X.ndim != 2 or X.shape[1] <= limit:
        return X
    finite = np.where(np.isfinite(X), X, 0.0)
    variances = np.var(finite, axis=0, dtype=np.float64)
    order = np.argsort(-variances, kind="stable")[:limit]
    return X[:, order]


def _target_metaspot_counts_from_site_counts(
    site_counts: np.ndarray,
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> np.ndarray:
    """Vectorize the configured MetaSpot budget over per-cell site counts."""
    n_sites = np.asarray(site_counts, dtype=np.int64)
    if np.any(n_sites < 0):
        raise ValueError("site_counts must be non-negative")

    target_sites = max(1, int(target_sites_per_metaspot))
    target_count = (n_sites + target_sites - 1) // target_sites
    lower_bound = max(1, int(min_metaspots_per_cell))
    target_count = np.maximum(lower_bound, target_count)
    if isinstance(max_metaspots_per_cell, (int, float)):
        upper_bound = max(lower_bound, int(max_metaspots_per_cell))
        target_count = np.minimum(target_count, upper_bound)
    target_count = np.minimum(target_count, n_sites)
    return np.maximum(1, target_count).astype(np.int64, copy=False)


def _target_metaspot_count(
    coordinates: np.ndarray,
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> int:
    """Choose a per-cell MetaSpot count from the number of unique XY sites."""
    xy = np.asarray(coordinates, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_points, 2)")
    finite_xy = xy[np.isfinite(xy).all(axis=1)]
    n_sites = int(np.unique(finite_xy, axis=0).shape[0])
    return int(
        _target_metaspot_counts_from_site_counts(
            np.asarray([n_sites], dtype=np.int64),
            target_sites_per_metaspot=target_sites_per_metaspot,
            min_metaspots_per_cell=min_metaspots_per_cell,
            max_metaspots_per_cell=max_metaspots_per_cell,
        )[0]
    )


def _allocate_partition_target_counts(
    partition_coordinates: Sequence[np.ndarray],
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> List[int]:
    """Allocate one cell-level MetaSpot budget across spatial compartments.

    Every non-empty compartment receives at least one MetaSpot.  The remaining
    budget is assigned in proportion to unique XY sites, with deterministic
    largest-remainder tie breaking.  This preserves the configured per-cell
    cap while preventing any MetaSpot from crossing a compartment boundary.
    """
    partitions = [
        np.asarray(coordinates, dtype=np.float32)
        for coordinates in partition_coordinates
        if len(coordinates)
    ]
    if not partitions:
        return []

    site_counts = np.asarray(
        [
            np.unique(
                coordinates[np.isfinite(coordinates).all(axis=1)],
                axis=0,
            ).shape[0]
            for coordinates in partitions
        ],
        dtype=np.int64,
    )
    site_counts = np.maximum(site_counts, 1)
    combined = np.vstack(partitions)
    total_target = _target_metaspot_count(
        combined,
        target_sites_per_metaspot=target_sites_per_metaspot,
        min_metaspots_per_cell=max(
            int(min_metaspots_per_cell),
            len(partitions),
        ),
        max_metaspots_per_cell=max_metaspots_per_cell,
    )
    total_target = max(len(partitions), int(total_target))
    total_target = min(int(site_counts.sum()), total_target)

    raw = total_target * site_counts.astype(np.float64) / site_counts.sum()
    allocation = np.floor(raw).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, site_counts)

    while int(allocation.sum()) < total_target:
        candidates = np.flatnonzero(allocation < site_counts)
        if candidates.size == 0:
            break
        residual = raw[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmax(residual)])
        allocation[chosen] += 1

    while int(allocation.sum()) > total_target:
        candidates = np.flatnonzero(allocation > 1)
        if candidates.size == 0:
            break
        residual = raw[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmin(residual)])
        allocation[chosen] -= 1

    return allocation.astype(int).tolist()


def _enforce_spatially_connected_labels(
    coordinates: np.ndarray,
    labels: np.ndarray,
    feature_matrix: Optional[np.ndarray] = None,
    target_n_clusters: Optional[int] = None,
    spatial_compactness: float = 1.0,
    connectivity_k: int = 8,
    radius_quantile: float = 0.9,
    radius_factor: float = 1.5,
    smoothing_iterations: int = 1,
    smoothing_min_fraction: float = 0.6,
    min_component_size: int = 20,
) -> np.ndarray:
    """Turn feature-space labels into spatially connected 2D components.

    Duplicate transcript coordinates are collapsed to one spatial site, labels
    are re-clustered with an XY-constrained Ward model when features are given,
    and undersized components are merged across a shared boundary. Without a
    feature matrix, the preliminary labels are smoothed and split directly.
    Component sizes are measured in transcripts, not unique XY sites.
    """
    xy = np.asarray(coordinates, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)
    n_points = len(point_labels)
    if n_points == 0:
        return point_labels.copy()
    if xy.ndim != 2 or xy.shape != (n_points, 2):
        raise ValueError("coordinates must have shape (n_points, 2)")
    features = None
    if feature_matrix is not None:
        features = np.asarray(feature_matrix, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != n_points:
            raise ValueError("feature_matrix must have n_points rows")

    finite = np.isfinite(xy).all(axis=1)
    if not finite.all():
        # Module 3 already drops invalid XY rows; keep this helper defensive.
        out = point_labels.copy()
        if finite.any():
            out[finite] = _enforce_spatially_connected_labels(
                xy[finite],
                point_labels[finite],
                feature_matrix=features[finite] if features is not None else None,
                target_n_clusters=target_n_clusters,
                spatial_compactness=spatial_compactness,
                connectivity_k=connectivity_k,
                radius_quantile=radius_quantile,
                radius_factor=radius_factor,
                smoothing_iterations=smoothing_iterations,
                smoothing_min_fraction=smoothing_min_fraction,
                min_component_size=min_component_size,
            )
        return out

    unique_xy, inverse, site_weights = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    n_sites = len(unique_xy)
    if n_sites <= 1:
        return np.zeros(n_points, dtype=np.int32)

    # Canonical spatial Ward segmentation starts from one constant label. Skip
    # thousands of tiny majority-vote reductions in that common path.
    point_label_values = np.unique(point_labels)
    if point_label_values.size == 1:
        site_labels = np.full(
            n_sites,
            int(point_label_values[0]),
            dtype=np.int32,
        )
    else:
        # Pick a deterministic majority label at coordinates containing
        # multiple transcripts.
        site_labels = np.empty(n_sites, dtype=np.int32)
        order = np.argsort(inverse, kind="stable")
        sorted_sites = inverse[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_sites)) + 1]
        stops = np.r_[starts[1:], len(order)]
        for start, stop in zip(starts, stops):
            site = int(sorted_sites[start])
            values, counts = np.unique(
                point_labels[order[start:stop]],
                return_counts=True,
            )
            site_labels[site] = int(values[np.argmax(counts)])

    query_k = max(2, min(n_sites, int(connectivity_k) + 1))
    neighbor_idx, neighbor_dist = _kdtree_query(unique_xy, query_k)
    if neighbor_idx is None or neighbor_dist is None:
        return point_labels.copy()

    positive_dist = np.where(neighbor_dist > 1e-8, neighbor_dist, np.inf)
    nearest = positive_dist.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return point_labels.copy()

    q = float(np.clip(radius_quantile, 0.0, 1.0))
    base_spacing = float(np.quantile(nearest, q))
    radius = base_spacing * max(float(radius_factor), 1e-6)
    if not np.isfinite(radius) or radius <= 0:
        return point_labels.copy()

    edge_rows = np.repeat(
        np.arange(n_sites, dtype=np.int64),
        neighbor_idx.shape[1],
    )
    edge_cols = np.asarray(neighbor_idx, dtype=np.int64).ravel()
    edge_distances = np.asarray(neighbor_dist, dtype=np.float64).ravel()
    valid_edges = (
        (edge_rows != edge_cols)
        & np.isfinite(edge_distances)
        & (edge_distances > 0.0)
    )
    edge_rows = edge_rows[valid_edges]
    edge_cols = edge_cols[valid_edges]
    edge_distances = edge_distances[valid_edges]
    edge_low = np.minimum(edge_rows, edge_cols)
    edge_high = np.maximum(edge_rows, edge_cols)
    edge_codes = edge_low * np.int64(n_sites) + edge_high
    knn_codes = np.unique(edge_codes)
    local_codes = np.unique(edge_codes[edge_distances <= radius])

    # Feature-aware Ward clustering needs one connected neighborhood graph to
    # preserve its requested cluster count. Direct label splitting keeps the
    # stricter native-spacing radius so genuinely separated islands stay split.
    selected_codes = knn_codes if features is not None else local_codes
    if selected_codes.size == 0:
        return point_labels.copy()

    edge_a = (selected_codes // np.int64(n_sites)).astype(
        np.int32,
        copy=False,
    )
    edge_b = (selected_codes % np.int64(n_sites)).astype(
        np.int32,
        copy=False,
    )

    used_constrained_clustering = False
    if features is not None and n_sites > 1:
        n_features = features.shape[1]
        site_features = np.zeros((n_sites, n_features), dtype=np.float32)
        np.add.at(site_features, inverse, features)
        site_features /= site_weights[:, None].astype(np.float32)
        site_features, _ = _safe_standardize(site_features)
        xy_features, _ = _safe_standardize(unique_xy)
        combined_features = np.hstack(
            [
                site_features,
                xy_features * max(float(spatial_compactness), 0.0),
            ]
        ).astype(np.float32, copy=False)

        requested_clusters = (
            int(target_n_clusters)
            if isinstance(target_n_clusters, (int, float))
            else int(np.unique(point_labels).size)
        )
        requested_clusters = max(1, min(requested_clusters, n_sites))
        if requested_clusters > 1:
            rows = np.concatenate((edge_a, edge_b))
            cols = np.concatenate((edge_b, edge_a))
            connectivity = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_sites, n_sites),
            )
            try:
                # Some runners prepend their own warning filters after this
                # module is imported, so reassert the precise rule at the call
                # site without wrapping or serializing parallel clustering.
                _quiet_expected_disconnected_connectivity_warning()
                site_labels = AgglomerativeClustering(
                    n_clusters=requested_clusters,
                    linkage="ward",
                    connectivity=connectivity,
                    compute_full_tree=False,
                ).fit_predict(combined_features).astype(np.int32)
                used_constrained_clustering = True
            except Exception as exc:
                _logging.getLogger("cellscope").warning(
                    "Spatially constrained clustering failed; using label "
                    "smoothing fallback: %s",
                    exc,
                )

    if not used_constrained_clustering:
        adjacency = [[] for _ in range(n_sites)]
        for a, b in zip(edge_a, edge_b):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        smooth_fraction = float(np.clip(smoothing_min_fraction, 0.5, 1.0))
        for _ in range(max(0, int(smoothing_iterations))):
            updated = site_labels.copy()
            changed = False
            for i, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                votes: Dict[int, int] = defaultdict(int)
                votes[int(site_labels[i])] += int(site_weights[i])
                for j in neighbors:
                    votes[int(site_labels[j])] += int(site_weights[j])
                current = int(site_labels[i])
                winner = max(
                    votes,
                    key=lambda label: (votes[label], label == current, -int(label)),
                )
                total = sum(votes.values())
                if (
                    winner != current
                    and total > 0
                    and votes[winner] / total >= smooth_fraction
                ):
                    updated[i] = int(winner)
                    changed = True
            site_labels = updated
            if not changed:
                break

    parent = np.arange(n_sites, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return int(node)

    def union(a: int, b: int) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        return True

    # First split every original label into its XY-connected components.
    for a, b in zip(edge_a, edge_b):
        if site_labels[a] == site_labels[b]:
            union(int(a), int(b))

    min_size = max(1, int(min_component_size))
    for _ in range(32):
        roots = np.asarray([find(i) for i in range(n_sites)], dtype=np.int32)
        component_size: Dict[int, int] = defaultdict(int)
        for site, root in enumerate(roots):
            component_size[int(root)] += int(site_weights[site])

        small_roots = {
            root for root, size in component_size.items() if size < min_size
        }
        if not small_roots:
            break

        boundary_contacts: Dict[int, Counter] = defaultdict(Counter)
        for a, b in zip(edge_a, edge_b):
            root_a, root_b = find(int(a)), find(int(b))
            if root_a == root_b:
                continue
            boundary_contacts[root_a][root_b] += 1
            boundary_contacts[root_b][root_a] += 1

        changed = False
        for root in sorted(small_roots, key=lambda r: (component_size[r], r)):
            root = find(root)
            if component_size.get(root, 0) >= min_size:
                continue
            candidates = {
                find(candidate): contacts
                for candidate, contacts in boundary_contacts.get(root, {}).items()
                if find(candidate) != root
            }
            if not candidates:
                continue
            target = max(
                candidates,
                key=lambda candidate: (
                    candidates[candidate],
                    component_size.get(candidate, 0),
                    -int(candidate),
                ),
            )
            changed |= union(root, target)
        if not changed:
            break

    roots = np.asarray([find(i) for i in range(n_sites)], dtype=np.int32)
    root_order = sorted(np.unique(roots), key=lambda root: int(np.flatnonzero(roots == root)[0]))
    root_to_label = {int(root): label for label, root in enumerate(root_order)}
    connected_site_labels = np.asarray(
        [root_to_label[int(root)] for root in roots],
        dtype=np.int32,
    )
    return connected_site_labels[inverse]


def per_cell_clustering(
    point_df: pd.DataFrame,
    gmm_cols: List[str],
    composition_model: Optional[_LocalCompositionModel] = None,
    show_internal_progress: bool = False,
    hdbscan_min_cluster_size: int = 6,
    hdbscan_allow_single_cluster: bool = True,
    hdbscan_min_samples_frac: float = 0.3,
    target_sub_size: int = 12,
    n_feat_hdb: int = 24,
    kmeans_min_k: int = 2,
    kmeans_max_k: int = 999,
    use_k_cap: bool = False,
    merge_min_size: Optional[int] = None,
    hdbscan_max_points: int = 512,
    kmeans_train_size: int = 1024,
    kmeans_batch_size: int = 2048,
    small_backend_cells: int = 12000,
    sequential_cells: int = 256,
    enforce_spatial_connectivity: bool = False,
    partition_by_nucleus: bool = False,
    spatial_connectivity_k: int = 8,
    spatial_radius_quantile: float = 0.9,
    spatial_radius_factor: float = 1.5,
    spatial_smoothing_iterations: int = 1,
    spatial_smoothing_min_fraction: float = 0.6,
    spatial_min_component_size: Optional[int] = None,
    spatial_compactness: float = 1.0,
    target_sites_per_metaspot: Optional[int] = None,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
    parallel_chunk_cells: Optional[int] = None,
    parallel_n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
    MIN_SUB_PTS = int(hdbscan_min_cluster_size)
    # 使用更小的目标簇大小以促进细分，且不与最小簇大小绑定
    TARGET_SUB_SIZE = max(8, int(target_sub_size))
    K_CAP_BASE = 999
    MIN_SAMPLES_FRAC = float(hdbscan_min_samples_frac)
    SPATIAL_MIN_COMPONENT_SIZE = (
        int(spatial_min_component_size)
        if isinstance(spatial_min_component_size, (int, float))
        else int(merge_min_size if merge_min_size is not None else MIN_SUB_PTS)
    )
    TARGET_SITES_PER_METASPOT = (
        int(target_sites_per_metaspot)
        if isinstance(target_sites_per_metaspot, (int, float))
        and int(target_sites_per_metaspot) > 0
        else None
    )

    def _finalize_spatial_labels(
        g: pd.DataFrame,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
        target_count_override: Optional[int] = None,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32)
        if not bool(enforce_spatial_connectivity) or len(labels) <= 1:
            return labels
        coordinates = g[["x_location", "y_location"]].to_numpy(
            np.float32,
            copy=False,
        )
        target_count = int(np.unique(labels).size)
        if isinstance(target_count_override, (int, float)):
            target_count = max(1, int(target_count_override))
        elif TARGET_SITES_PER_METASPOT is not None:
            target_count = _target_metaspot_count(
                coordinates,
                target_sites_per_metaspot=TARGET_SITES_PER_METASPOT,
                min_metaspots_per_cell=min_metaspots_per_cell,
                max_metaspots_per_cell=max_metaspots_per_cell,
            )
        return _enforce_spatially_connected_labels(
            coordinates,
            labels,
            feature_matrix=features,
            target_n_clusters=target_count,
            spatial_compactness=spatial_compactness,
            connectivity_k=spatial_connectivity_k,
            radius_quantile=spatial_radius_quantile,
            radius_factor=spatial_radius_factor,
            smoothing_iterations=spatial_smoothing_iterations,
            smoothing_min_fraction=spatial_smoothing_min_fraction,
            min_component_size=SPATIAL_MIN_COMPONENT_SIZE,
        )

    # Select features with the highest variance. If per-row composition columns
    # are not materialized in `point_df` (to avoid large memory), a
    # precomputed per-cell composition DataFrame may be present in the
    # environment and will be used when building per-transcript features.
    # We build a modest sample by combining per-row geometry and per-cell
    # composition (looked up) to compute variances without expanding global
    # comp arrays.
    geom_cols = [c for c in gmm_cols if c in ('x_location', 'y_location', 'r_signed', 'd_cell_norm', 'd_nuc_norm', 'sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta', 'local_density')]
    comp_cols = [c for c in gmm_cols if c not in geom_cols]

    def _decide_K_min(n: int) -> int:
        # 更积极的最低簇数下界
        if n >= 6 * MIN_SUB_PTS:
            return 4
        elif n >= 4 * MIN_SUB_PTS:
            return 3
        elif n >= 3 * MIN_SUB_PTS:
            return 2
        else:
            return 1

    def _decide_K_for_cell(n: int) -> int:
        if n < MIN_SUB_PTS * 1.2:
            return 1
        k_by_size = max(1, n // MIN_SUB_PTS)
        k_target = max(1, int(np.ceil(n / float(TARGET_SUB_SIZE))))
        k_cap = min(K_CAP_BASE, k_by_size)
        K = min(k_target, k_cap)
        K = max(_decide_K_min(n), K)
        if n >= 8 * MIN_SUB_PTS:
            K = max(K, _decide_K_min(n) + 1)
        # Optional clamp by configured bounds
        if bool(use_k_cap):
            K = max(int(kmeans_min_k), min(int(kmeans_max_k), int(K)))
        return int(max(1, K))

    def _merge_small_clusters(labels: np.ndarray, X: np.ndarray, min_size: int) -> np.ndarray:
        y = labels.astype(np.int32, copy=True)
        while True:
            labs, cnts = np.unique(y, return_counts=True)
            tiny = labs[cnts < min_size]
            if len(tiny) == 0:
                break
            big = labs[cnts >= min_size]
            if len(big) == 0:
                big = np.array([labs[0]], dtype=labs.dtype)
            cents = {lab: X[y == lab].mean(0) for lab in labs}
            big_cent = np.vstack([cents[b] for b in big])
            for t in tiny:
                idx = np.where(y == t)[0]
                if idx.size == 0:
                    continue
                c_t = cents[t][None, :]
                from scipy.spatial.distance import cdist
                j = np.argmin(cdist(c_t, big_cent))
                to_lab = big[j]
                y[idx] = to_lab
            u = np.unique(y)
            remap = {u_i: i for i, u_i in enumerate(u)}
            y = np.vectorize(remap.get, otypes=[np.int32])(y)
        return y

    def _assign_noise_to_nearest(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        y = labels.astype(np.int32, copy=True)
        mask_noise = (y < 0)
        if not mask_noise.any():
            return y
        labs = np.unique(y[~mask_noise])
        if labs.size == 0:
            return y
        cents = np.vstack([X[y == lab].mean(0) for lab in labs])
        idx_noise = np.where(mask_noise)[0]
        X_noise = X[idx_noise]
        from scipy.spatial.distance import cdist
        D = cdist(X_noise, cents)
        nearest = labs[D.argmin(axis=1)]
        y[idx_noise] = nearest
        u = np.unique(y)
        remap = {u_i: i for i, u_i in enumerate(u)}
        y = np.vectorize(remap.get, otypes=[np.int32])(y)
        return y

    # Per-cell MetaSpot segmentation (parallel batching)

    def _iter_group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    n_groups = int(point_df['cell_id'].nunique(dropna=False)) if len(point_df) else 0

    # Deterministic RNG for MiniBatchKMeans sampling inside _fit_one
    rng = np.random.default_rng(42)
    # Keep process workers from serializing the full transcript table.
    cell_comp_emb = getattr(point_df, '_cell_comp_emb', None)

    def _fit_one(
        args,
        allow_nucleus_partition: bool = True,
        target_count_override: Optional[int] = None,
    ):
        cid, g = args
        # Ensure minimal geometry present; composition may be provided per-cell
        g = g.dropna(subset=['x_location', 'y_location'])
        if len(g) == 0:
            return cid, pd.Series([], dtype=np.int32, index=g.index)

        if bool(partition_by_nucleus) and allow_nucleus_partition:
            if "in_nucleus" in g.columns:
                nucleus_flags = _coerce_bool_series(
                    g["in_nucleus"],
                    default=False,
                )
            elif "r_signed" in g.columns:
                nucleus_flags = (
                    pd.to_numeric(g["r_signed"], errors="coerce")
                    .fillna(0.0)
                    .lt(0.0)
                )
            else:
                raise ValueError(
                    "module3.partition_by_nucleus requires either "
                    "'in_nucleus' or 'r_signed' in the Module 3 point table"
                )

            present_partitions = [
                flag
                for flag in (True, False)
                if bool(nucleus_flags.eq(flag).any())
            ]
            if len(present_partitions) > 1:
                partition_frames = [
                    g.loc[nucleus_flags.eq(flag)]
                    for flag in present_partitions
                ]
                if TARGET_SITES_PER_METASPOT is not None:
                    partition_targets = _allocate_partition_target_counts(
                        [
                            frame[
                                ["x_location", "y_location"]
                            ].to_numpy(np.float32, copy=False)
                            for frame in partition_frames
                        ],
                        target_sites_per_metaspot=TARGET_SITES_PER_METASPOT,
                        min_metaspots_per_cell=min_metaspots_per_cell,
                        max_metaspots_per_cell=max_metaspots_per_cell,
                    )
                else:
                    partition_targets = [None] * len(partition_frames)

                combined = pd.Series(
                    index=g.index,
                    dtype=np.int32,
                )
                label_offset = 0
                for frame, partition_target in zip(
                    partition_frames,
                    partition_targets,
                ):
                    _, partition_labels = _fit_one(
                        (cid, frame),
                        allow_nucleus_partition=False,
                        target_count_override=partition_target,
                    )
                    dense_labels = pd.factorize(
                        partition_labels,
                        sort=True,
                    )[0].astype(np.int32)
                    combined.loc[partition_labels.index] = (
                        dense_labels + label_offset
                    )
                    label_offset += max(
                        1,
                        int(np.unique(dense_labels).size),
                    )
                return cid, combined.astype(np.int32)

        n = len(g)
        if n < MIN_SUB_PTS:
            labels = np.zeros(n, dtype=np.int32)
            labels = _finalize_spatial_labels(
                g,
                labels,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        # Build local expression features for this cell only, then release them
        # after clustering. This avoids materializing dense embeddings globally.
        local_density = None
        if composition_model is not None:
            X_comp, local_density = composition_model.transform_cell(g)
        else:
            comp_present = any((cc in g.columns) for cc in comp_cols)
        if composition_model is None and comp_present:
            X_comp = g[[c for c in comp_cols if c in g.columns]].to_numpy(np.float32, copy=False)
            # If some comp_cols are missing, pad zeros
            if X_comp.shape[1] < len(comp_cols):
                missing = len(comp_cols) - X_comp.shape[1]
                X_comp = np.hstack([X_comp, np.zeros((len(g), missing), dtype=np.float32)])
        elif composition_model is None:
            if cell_comp_emb is not None and cid in cell_comp_emb.index:
                vec = cell_comp_emb.loc[cid, comp_cols].to_numpy(dtype=np.float32)
                X_comp = np.repeat(vec.reshape(1, -1), len(g), axis=0)
            else:
                X_comp = np.zeros((len(g), len(comp_cols)), dtype=np.float32)
        # geometry
        geom_use = [c for c in geom_cols if c in g.columns and c != 'local_density']
        if len(geom_use) == 0:
            X_geom = np.zeros((len(g), 2), dtype=np.float32)
        else:
            X_geom = g[geom_use].to_numpy(np.float32, copy=False)
        if local_density is not None:
            X_geom = np.column_stack((X_geom, local_density)).astype(np.float32, copy=False)
        X_full = np.hstack([X_geom, X_comp]) if X_comp.size else X_geom
        X_full = _select_top_variance_features(X_full, n_feat_hdb)
        Xz, good = _safe_standardize(X_full)
        if Xz.shape[1] == 0 or not np.any(good):
            X_coord = g[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
            Xz, _ = _safe_standardize(X_coord)
        # In the canonical MetaSpot mode, spatial resolution determines K.
        # HDBSCAN remains available as a legacy fallback when this target is off.
        if bool(enforce_spatial_connectivity) and TARGET_SITES_PER_METASPOT is not None:
            seed_labels = np.zeros(n, dtype=np.int32)
            labels = _finalize_spatial_labels(
                g,
                seed_labels,
                Xz,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        # Fast path: large cells switch to sampled MiniBatchKMeans to avoid expensive HDBSCAN
        if n > int(hdbscan_max_points):
            K_init = _decide_K_for_cell(n)
            if bool(use_k_cap):
                K_init = max(int(kmeans_min_k), min(int(kmeans_max_k), int(K_init)))
            K_init = max(1, int(K_init))
            train_n = min(int(kmeans_train_size), n)
            K_init = min(K_init, max(1, train_n))
            idx_train = rng.choice(n, size=train_n, replace=False)
            X_train = Xz[idx_train]
            bs = max(64, min(int(kmeans_batch_size), train_n))
            km = MiniBatchKMeans(
                n_clusters=K_init,
                init="k-means++",
                n_init=3,
                random_state=42,
                batch_size=bs,
                max_iter=80,
                reassignment_ratio=0.01,
                verbose=0,
            )
            km.fit(X_train)
            labels = km.predict(Xz).astype(np.int32)
            _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
            labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
            labels = _finalize_spatial_labels(
                g,
                labels,
                Xz,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)

        min_samples = max(3, int(round(MIN_SUB_PTS * MIN_SAMPLES_FRAC)))
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_SUB_PTS,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=1,
            prediction_data=False,
            allow_single_cluster=bool(hdbscan_allow_single_cluster),
            gen_min_span_tree=False,
            approx_min_span_tree=False,
        )
        hdb_labels = clusterer.fit_predict(Xz).astype(np.int32)
        mask_non_noise = (hdb_labels >= 0)
        uniq_clusters = np.unique(hdb_labels[mask_non_noise]) if mask_non_noise.any() else np.array([], dtype=int)
        n_clusters = len(uniq_clusters)
        K_min = max(int(kmeans_min_k), _decide_K_min(n))
        use_fallback = False
        if n_clusters == 0:
            use_fallback = True
        elif (n_clusters < K_min) and (n >= int(1.5 * MIN_SUB_PTS)):
            use_fallback = True
        if use_fallback:
            K_init = _decide_K_for_cell(n)
            if K_init <= 1:
                labels = np.zeros(n, dtype=np.int32)
            else:
                km = KMeans(n_clusters=K_init, n_init='auto', random_state=42, max_iter=200)
                labels = km.fit_predict(Xz).astype(np.int32)
            _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
            labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
            labels = _finalize_spatial_labels(
                g,
                labels,
                Xz,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        labels = _assign_noise_to_nearest(hdb_labels, Xz)
        _min_size = int(merge_min_size) if merge_min_size is not None else int(MIN_SUB_PTS)
        labels = _merge_small_clusters(labels, Xz, min_size=_min_size)
        labels = _finalize_spatial_labels(
            g,
            labels,
            Xz,
            target_count_override=target_count_override,
        )
        return cid, pd.Series(labels, index=g.index, dtype=np.int32)

    # Lightweight backend selection for small datasets to cut scheduler/IPC overhead
    _force_threads = bool(os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip())
    use_sequential = n_groups <= int(sequential_cells)
    _backend = _joblib_backend()
    if _force_threads or n_groups <= int(small_backend_cells):
        _backend = "threading"
    if use_sequential:
        iterator = _progress_iter(range(n_groups), desc='MetaSpot segmentation (seq)', total=n_groups, show=show_internal_progress)
        labels_per_cell = pd.Series(
            -1,
            index=point_df.index,
            dtype=np.int32,
        )
        for chunk in _iter_group_chunks(point_df, 1):
            for pair in chunk:
                _, labels = _fit_one(pair)
                labels_per_cell.loc[labels.index] = labels.to_numpy(
                    dtype=np.int32,
                    copy=False,
                )
        point_df = point_df.copy(deep=False)
        point_df[METASPOT_LOCAL_COL] = labels_per_cell
        missing = point_df[METASPOT_LOCAL_COL].isna()
        if missing.any():
            point_df.loc[missing, METASPOT_LOCAL_COL] = -1
        point_df[METASPOT_LOCAL_COL] = point_df[METASPOT_LOCAL_COL].astype(np.int32)
        point_df[METASPOT_COL] = (
            point_df['cell_id'].astype(str)
            + ':'
            + point_df[METASPOT_LOCAL_COL].astype(str)
        )
        return point_df

    from joblib import Parallel, delayed
    # Ward/k-d tree work releases the GIL and each cell is bounded in size.
    # Use several cells concurrently by default while keeping an explicit
    # upper bound for large production runs.
    available_jobs = _preferred_n_jobs()
    if isinstance(parallel_n_jobs, (int, float)) and int(parallel_n_jobs) > 0:
        N_JOBS = max(1, min(int(parallel_n_jobs), available_jobs))
    else:
        N_JOBS = max(1, min(8, available_jobs))
    # _backend already chosen above, but keep env override precedence
    if os.environ.get("CELLSCOPE_FORCE_THREADS", "").strip():
        _backend = "threading"

    # prefer larger chunk to reduce scheduler overhead
    chunk_size = max(64, _cluster_chunk_size(parallel_chunk_cells))
    labels_per_cell = pd.Series(
        -1,
        index=point_df.index,
        dtype=np.int32,
    )
    n_chunks = (n_groups + chunk_size - 1) // chunk_size if n_groups else 0
    iterator = _progress_iter(
        _iter_group_chunks(point_df, chunk_size),
        desc='MetaSpot segmentation (parallel)',
        total=n_chunks,
        show=show_internal_progress,
    )
    _logging.getLogger("cellscope").info(
        "Module 3 parallel segmentation | cells=%d | workers=%d | backend=%s | chunk_cells=%d",
        n_groups,
        N_JOBS,
        _backend,
        chunk_size,
    )
    # Keep one executor for all chunks.  Recreating it for every small chunk
    # dominated scheduling time on large COAD runs.
    from contextlib import nullcontext

    thread_limit = nullcontext()
    if N_JOBS > 1:
        try:
            from threadpoolctl import threadpool_limits

            # Prevent every concurrently segmented cell from starting another
            # BLAS/OpenMP pool. This context is process-wide and therefore
            # belongs outside the persistent joblib executor.
            thread_limit = threadpool_limits(limits=1)
        except Exception:
            thread_limit = nullcontext()
    with thread_limit:
        with Parallel(
            n_jobs=N_JOBS,
            backend=_backend,
            verbose=0,
            pre_dispatch=f"3*{N_JOBS}",
            batch_size="auto",
            max_nbytes=None,
        ) as parallel:
            for chunk_index, chunk in enumerate(iterator, start=1):
                try:
                    out = parallel(delayed(_fit_one)(pair) for pair in chunk)
                except Exception as e:
                    # Fallback: sequential processing to avoid hard failure
                    # (for example, a worker memory allocation failure).
                    print(
                        "[WARN] Parallel cluster chunk failed, fallback "
                        f"sequential: {e}"
                    )
                    out = [_fit_one(pair) for pair in chunk]
                if out:
                    chunk_labels = pd.concat(
                        [series for _, series in out]
                    ).astype(np.int32, copy=False)
                    labels_per_cell.loc[
                        chunk_labels.index
                    ] = chunk_labels.to_numpy(
                        dtype=np.int32,
                        copy=False,
                    )
                del out
                # Periodic collection retains bounded temporary lifetimes
                # without making collection the primary segmentation cost.
                if chunk_index % 16 == 0:
                    gc.collect()
    gc.collect()

    point_df = point_df.copy(deep=False)
    point_df[METASPOT_LOCAL_COL] = labels_per_cell
    missing = point_df[METASPOT_LOCAL_COL].isna()
    if missing.any():
        point_df.loc[missing, METASPOT_LOCAL_COL] = -1
    point_df[METASPOT_LOCAL_COL] = point_df[METASPOT_LOCAL_COL].astype(np.int32)
    point_df[METASPOT_COL] = (
        point_df['cell_id'].astype(str)
        + ':'
        + point_df[METASPOT_LOCAL_COL].astype(str)
    )
    return point_df


def _metaspot_segmentation_audit(
    point_df: pd.DataFrame,
    *,
    target_sites_per_metaspot: Optional[int] = None,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
    partition_by_nucleus: bool = False,
) -> pd.DataFrame:
    """Return compact per-cell checks for MetaSpot size and compartment mixing."""
    required = {
        "cell_id",
        METASPOT_COL,
        "x_location",
        "y_location",
    }
    missing = sorted(required.difference(point_df.columns))
    if missing:
        raise KeyError(
            "MetaSpot segmentation audit requires columns: "
            + ", ".join(missing)
        )

    audit_source = point_df[
        [
            "cell_id",
            METASPOT_COL,
            "x_location",
            "y_location",
        ]
    ].copy()
    if "in_nucleus" in point_df.columns:
        audit_source["_in_nucleus"] = _coerce_bool_series(
            point_df["in_nucleus"],
            default=False,
        ).to_numpy()
    elif "r_signed" in point_df.columns:
        audit_source["_in_nucleus"] = (
            pd.to_numeric(point_df["r_signed"], errors="coerce")
            .fillna(0.0)
            .lt(0.0)
            .to_numpy()
        )
    else:
        audit_source["_in_nucleus"] = False

    grouped_spot = audit_source.groupby(
        ["cell_id", METASPOT_COL],
        sort=False,
        observed=True,
    )
    spot_stats = grouped_spot.agg(
        n_transcripts=(METASPOT_COL, "size"),
        nucleus_min=("_in_nucleus", "min"),
        nucleus_max=("_in_nucleus", "max"),
    )
    unique_sites = (
        audit_source[
            ["cell_id", METASPOT_COL, "x_location", "y_location"]
        ].drop_duplicates(
            ["cell_id", METASPOT_COL, "x_location", "y_location"]
        )
        .groupby(
            ["cell_id", METASPOT_COL],
            sort=False,
            observed=True,
        )
        .size()
        .rename("n_unique_sites")
    )
    spot_stats = spot_stats.join(unique_sites, how="left")
    spot_stats["nucleus_mixed"] = (
        spot_stats["nucleus_min"] != spot_stats["nucleus_max"]
    )
    spot_stats["mixed_transcripts"] = (
        spot_stats["n_transcripts"]
        * spot_stats["nucleus_mixed"].astype(np.int64)
    )

    per_cell = spot_stats.groupby(
        level="cell_id",
        sort=False,
        observed=True,
    ).agg(
        actual_metaspots=("n_transcripts", "size"),
        median_transcripts_per_metaspot=("n_transcripts", "median"),
        max_transcripts_per_metaspot=("n_transcripts", "max"),
        median_sites_per_metaspot=("n_unique_sites", "median"),
        max_sites_per_metaspot=("n_unique_sites", "max"),
        nucleus_mixed_metaspots=("nucleus_mixed", "sum"),
        mixed_transcripts=("mixed_transcripts", "sum"),
        n_transcripts=("n_transcripts", "sum"),
    )
    per_cell["nucleus_mixed_metaspot_fraction"] = (
        per_cell["nucleus_mixed_metaspots"]
        / per_cell["actual_metaspots"].clip(lower=1)
    )
    per_cell["nucleus_mixed_transcript_fraction"] = (
        per_cell["mixed_transcripts"]
        / per_cell["n_transcripts"].clip(lower=1)
    )

    unique_cell_sites = audit_source[
        ["cell_id", "x_location", "y_location"]
    ].drop_duplicates()
    site_counts = unique_cell_sites.groupby(
        "cell_id",
        sort=False,
        observed=True,
    ).size()
    per_cell["n_unique_sites"] = site_counts.reindex(per_cell.index).astype(int)
    if isinstance(target_sites_per_metaspot, (int, float)):
        unique_xy = unique_cell_sites[
            ["x_location", "y_location"]
        ].to_numpy(np.float32, copy=False)
        finite_site_mask = np.isfinite(unique_xy).all(axis=1)
        finite_site_counts = (
            unique_cell_sites.loc[finite_site_mask, ["cell_id"]]
            .groupby("cell_id", sort=False, observed=True)
            .size()
            .reindex(per_cell.index, fill_value=0)
            .to_numpy(dtype=np.int64, copy=False)
        )
        per_cell["requested_metaspots"] = (
            _target_metaspot_counts_from_site_counts(
                finite_site_counts,
                target_sites_per_metaspot=int(target_sites_per_metaspot),
                min_metaspots_per_cell=min_metaspots_per_cell,
                max_metaspots_per_cell=max_metaspots_per_cell,
            )
        )
    else:
        per_cell["requested_metaspots"] = np.nan
    per_cell["actual_to_requested_ratio"] = (
        per_cell["actual_metaspots"]
        / pd.to_numeric(
            per_cell["requested_metaspots"],
            errors="coerce",
        ).replace(0, np.nan)
    )
    per_cell["partition_by_nucleus"] = bool(partition_by_nucleus)
    return per_cell.reset_index()


def _grouped_nanmean_std(
    codes: np.ndarray,
    values: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute NaN-aware population moments for contiguous integer groups."""
    valid = np.isfinite(values)
    counts = np.bincount(codes[valid], minlength=n_groups).astype(
        np.int64,
        copy=False,
    )
    sums = np.bincount(
        codes[valid],
        weights=values[valid],
        minlength=n_groups,
    )
    sumsq = np.bincount(
        codes[valid],
        weights=np.square(values[valid], dtype=np.float64),
        minlength=n_groups,
    )
    mean = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    variance = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sumsq, counts, out=variance, where=counts > 0)
    variance -= np.square(mean)
    variance = np.maximum(variance, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _grouped_geometry_shape(
    codes: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_groups: int,
    *,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sample-covariance elongation and spread for each MetaSpot."""
    x_mean, _ = _grouped_nanmean_std(codes, x_values, n_groups)
    y_mean, _ = _grouped_nanmean_std(codes, y_values, n_groups)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    counts = np.bincount(codes[valid], minlength=n_groups).astype(
        np.int64,
        copy=False,
    )
    x = x_values[valid].astype(np.float64, copy=False)
    y = y_values[valid].astype(np.float64, copy=False)
    grouped = lambda weights: np.bincount(
        codes[valid], weights=weights, minlength=n_groups
    )
    sum_x = grouped(x)
    sum_y = grouped(y)
    sum_x2 = grouped(np.square(x))
    sum_y2 = grouped(np.square(y))
    sum_xy = grouped(x * y)
    mx = x_mean.astype(np.float64, copy=False)
    my = y_mean.astype(np.float64, copy=False)
    count_float = counts.astype(np.float64, copy=False)
    var_x = sum_x2 - 2.0 * mx * sum_x + count_float * np.square(mx)
    var_y = sum_y2 - 2.0 * my * sum_y + count_float * np.square(my)
    cov_xy = sum_xy - mx * sum_y - my * sum_x + count_float * mx * my
    denominator = np.maximum(count_float - 1.0, 1.0)
    var_x /= denominator
    var_y /= denominator
    cov_xy /= denominator
    trace = np.maximum(var_x + var_y, 0.0)
    discriminant = np.maximum(
        np.square(trace) - 4.0 * (var_x * var_y - np.square(cov_xy)),
        0.0,
    )
    root = np.sqrt(discriminant)
    lambda_1 = np.maximum(0.5 * (trace + root), eps)
    lambda_2 = np.maximum(0.5 * (trace - root), eps)
    elongation = np.full(n_groups, np.nan, dtype=np.float32)
    spread = np.full(n_groups, np.nan, dtype=np.float32)
    valid_shape = counts >= 3
    elongation[valid_shape] = (
        lambda_1[valid_shape] / lambda_2[valid_shape]
    ).astype(np.float32)
    spread[valid_shape] = np.sqrt(trace[valid_shape]).astype(np.float32)
    return elongation, spread


def build_adata1(point_df: pd.DataFrame, show_internal_progress: bool = False) -> ad.AnnData:
    metaspot_cat = pd.Categorical(point_df[METASPOT_COL], ordered=False)
    row_codes = metaspot_cat.codes.astype(np.int64)
    row_cats = metaspot_cat.categories
    gene_codes, gene_uni = pd.factorize(point_df['feature_name'], sort=False)
    gene_codes = gene_codes.astype(np.int64)
    X = sp.coo_matrix(
        (np.ones(len(point_df), dtype=np.int32), (row_codes, gene_codes)),
        shape=(len(row_cats), len(gene_uni))
    ).tocsr()
    del gene_codes
    adata1 = ad.AnnData(X=X, obs=pd.DataFrame(index=row_cats), var=pd.DataFrame(index=gene_uni))

    n_groups = len(row_cats)
    # Do not materialize a five-column, row-sorted transcript array.  The
    # COAD run contains hundreds of millions of rows, so that copy alone was
    # several GiB before any per-MetaSpot statistics were calculated.
    codes = row_codes.astype(
        np.int32 if n_groups <= np.iinfo(np.int32).max else np.int64,
        copy=False,
    )
    if codes is not row_codes:
        del row_codes
    counts = np.bincount(codes, minlength=n_groups).astype(np.int64, copy=False)

    r_values = point_df['r_signed'].to_numpy(dtype=np.float32, copy=False)
    r_mean, r_std = _grouped_nanmean_std(codes, r_values, n_groups)
    # Pandas executes grouped quantiles in native code.  This preserves the
    # exact NumPy interpolation rule used previously while avoiding one Python
    # percentile call per MetaSpot.
    quantiles = (
        pd.Series(r_values, copy=False)
        .groupby(codes, sort=True, observed=True)
        .quantile([0.1, 0.5, 0.9])
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_groups, 3)
    )
    r_p10, r_p50, r_p90 = quantiles.T
    del quantiles, r_values

    d1_values = point_df['d_cell_norm'].to_numpy(dtype=np.float32, copy=False)
    d1_mean, _ = _grouped_nanmean_std(codes, d1_values, n_groups)
    del d1_values
    d2_values = point_df['d_nuc_norm'].to_numpy(dtype=np.float32, copy=False)
    d2_mean, _ = _grouped_nanmean_std(codes, d2_values, n_groups)
    del d2_values
    x_values = point_df['x_location'].to_numpy(dtype=np.float32, copy=False)
    y_values = point_df['y_location'].to_numpy(dtype=np.float32, copy=False)
    elong, spread = _grouped_geometry_shape(
        codes,
        x_values,
        y_values,
        n_groups,
        eps=1e-8,
    )
    del x_values, y_values
    log_n = np.log1p(counts).astype(np.float32)

    md_stats = pd.DataFrame({
        METASPOT_COL: row_cats,
        'r_mean': r_mean, 'r_std': r_std,
        'r_p10': r_p10, 'r_p50': r_p50, 'r_p90': r_p90,
        'd_cell_norm_mean': d1_mean, 'd_nuc_norm_mean': d2_mean,
        'elongation': elong, 'spread': spread, 'log_n': log_n
    }).set_index(METASPOT_COL)
    adata1.obs = adata1.obs.join(md_stats, how='left')
    emb_cols = [c for c in point_df.columns if c.startswith('comp_emb')]
    if len(emb_cols):
        agg_comp = (
            point_df.groupby(METASPOT_COL)[emb_cols + ['local_density']]
            .agg(['mean', 'std'])
        )
        agg_comp.columns = [f"{a}_{b}" for a, b in agg_comp.columns]
        adata1.obs = adata1.obs.join(agg_comp, how='left')
    return adata1


def _attach_metaspot_spatial_metadata(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
) -> ad.AnnData:
    """Attach one owning cell and one physical centroid to every MetaSpot."""
    required = {METASPOT_COL, "x_location", "y_location"}
    missing = sorted(required.difference(point_df.columns))
    if missing:
        raise ValueError(
            "MetaSpot graph construction requires point_df columns "
            f"{missing}"
        )

    columns = [METASPOT_COL, "x_location", "y_location"]
    if "cell_id" in point_df.columns:
        columns.append("cell_id")
    frame = point_df.loc[:, columns].copy()
    frame[METASPOT_COL] = frame[METASPOT_COL].astype(str)

    grouped = frame.groupby(METASPOT_COL, observed=True, sort=False)
    centroids = grouped[["x_location", "y_location"]].mean()
    centroids.columns = ["x_centroid", "y_centroid"]

    if "cell_id" in frame.columns:
        owner_counts = grouped["cell_id"].nunique(dropna=False)
        if owner_counts.ne(1).any():
            bad = owner_counts.index[owner_counts.ne(1)].astype(str).tolist()[:5]
            raise ValueError(
                "Each MetaSpot must belong to exactly one cell; conflicting "
                f"owners found for {bad}"
            )
        owners = grouped["cell_id"].first().astype(str).rename("cell_id")
    elif "cell_id" in adata1.obs.columns:
        owners = adata1.obs["cell_id"].astype(str)
    else:
        raise ValueError(
            "MetaSpot graph construction requires cell_id in point_df or "
            "adata1.obs"
        )

    target = pd.Index(adata1.obs_names.astype(str))
    centroids.index = centroids.index.astype(str)
    owners.index = owners.index.astype(str)
    centroids = centroids.reindex(target)
    owners = owners.reindex(target)
    if centroids.isna().any(axis=None) or owners.isna().any():
        missing_ids = target[
            centroids.isna().any(axis=1).to_numpy() | owners.isna().to_numpy()
        ].tolist()[:5]
        raise ValueError(
            "Missing MetaSpot centroid or owner for observations "
            f"{missing_ids}"
        )

    adata1.obs = adata1.obs.copy()
    adata1.obs["cell_id"] = owners.to_numpy(dtype=str)
    adata1.obs["x_centroid"] = centroids["x_centroid"].to_numpy(np.float32)
    adata1.obs["y_centroid"] = centroids["y_centroid"].to_numpy(np.float32)
    return adata1


def _metaspot_expression_embedding(
    adata1: ad.AnnData,
    n_components: int = 32,
) -> np.ndarray:
    """Build a compact TF-IDF/SVD expression representation per MetaSpot."""
    X = (
        adata1.X.tocsr().astype(np.float32)
        if sp.issparse(adata1.X)
        else sp.csr_matrix(np.asarray(adata1.X, dtype=np.float32))
    )
    if X.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)

    row_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    row_sum[row_sum <= 0] = 1.0
    tf = X.multiply(1.0 / row_sum[:, None])
    gene_df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log1p(X.shape[0] / (1.0 + gene_df)).astype(np.float32)
    tfidf = tf.multiply(idf)

    max_components = min(
        max(1, int(n_components)),
        max(1, X.shape[0] - 1),
        max(1, X.shape[1] - 1),
    )
    if X.shape[0] == 1 or X.shape[1] == 1:
        latent = tfidf.toarray().astype(np.float32, copy=False)
    else:
        latent = TruncatedSVD(
            n_components=max_components,
            random_state=42,
            n_iter=5,
        ).fit_transform(tfidf).astype(np.float32)
    latent = StandardScaler().fit_transform(latent).astype(np.float32)
    latent[~np.isfinite(latent)] = 0.0
    return latent


def _symmetric_knn_connectivity(
    values: np.ndarray,
    n_neighbors: int,
) -> sp.csr_matrix:
    """Return a symmetric binary kNN graph with k clamped to data size."""
    from sklearn.neighbors import kneighbors_graph

    n_obs = int(values.shape[0])
    if n_obs <= 1 or int(n_neighbors) <= 0:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)
    k = min(max(1, int(n_neighbors)), n_obs - 1)
    # sklearn's brute-force distance kernel is already vectorized. Spawning a
    # joblib task per query chunk is slower for both the global graph and the
    # many small within-cell graphs, and can multiply BLAS pools. Keep the
    # exact graph deterministic and single-scheduled; BLAS/OpenMP remains
    # controlled by the process-level runtime settings.
    graph = kneighbors_graph(
        values,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
        n_jobs=1,
    ).astype(np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _within_cell_spatial_connectivity(
    coordinates: np.ndarray,
    cell_ids: Sequence[object],
    n_neighbors: int,
) -> sp.csr_matrix:
    """Build physical kNN edges, strictly within each owning cell."""
    n_obs = int(coordinates.shape[0])
    if n_obs == 0 or int(n_neighbors) <= 0:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)

    cells = np.asarray(cell_ids).astype(str)
    cell_codes, _ = pd.factorize(cells, sort=False)
    cell_codes = np.asarray(cell_codes, dtype=np.int64)
    order = np.argsort(cell_codes, kind="stable").astype(
        np.int64,
        copy=False,
    )
    counts = np.bincount(cell_codes).astype(np.int64, copy=False)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for group_index in range(counts.size):
        indices = order[offsets[group_index]:offsets[group_index + 1]]
        if indices.size <= 1:
            continue
        local = _symmetric_knn_connectivity(
            coordinates[indices],
            min(int(n_neighbors), indices.size - 1),
        ).tocoo()
        if local.nnz:
            rows.append(indices[local.row].astype(np.int64, copy=False))
            cols.append(indices[local.col].astype(np.int64, copy=False))

    if not rows:
        return sp.csr_matrix((n_obs, n_obs), dtype=np.float32)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.ones(row.size, dtype=np.float32)
    graph = sp.csr_matrix((data, (row, col)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _normalize_sparse_adjacency(
    adjacency: sp.csr_matrix,
    mode: str = "gcn",
) -> sp.csr_matrix:
    """Normalize an adjacency matrix after adding explicit self loops."""
    from scipy.sparse import diags, eye

    graph = adjacency.tocsr().astype(np.float32, copy=False)
    graph = graph + eye(graph.shape[0], dtype=np.float32, format="csr")
    graph.sum_duplicates()
    graph.eliminate_zeros()
    norm_mode = str(mode).strip().lower()
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    degree[degree <= 0] = 1.0
    if norm_mode == "gcn":
        scale = 1.0 / np.sqrt(degree)
        graph = (diags(scale) @ graph @ diags(scale)).tocsr()
    elif norm_mode == "rw":
        graph = (diags(1.0 / degree) @ graph).tocsr()
    elif norm_mode != "none":
        raise ValueError(
            "MetaSpot graph edge_norm must be one of {'gcn', 'rw', 'none'}"
        )
    graph = graph.astype(np.float32, copy=False)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def build_metaspot_graph(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    k_expr: int = 15,
    k_spatial: int = 6,
    expr_n_components: int = 32,
    alpha_expr: float = 0.8,
    alpha_spatial: float = 0.2,
    edge_norm: str = "gcn",
    use_geometry_features: bool = False,
    expression_cross_cell_only: bool = False,
    max_full_batch_nodes: int = 200_000,
    execution_profile: str = "exact",
    knn_backend: str = "exact",
    ann_candidate_multiplier: int = 4,
    ann_query_chunk_nodes: int = 100_000,
    ann_train_sample: int = 1_000_000,
    faiss_hnsw_m: int = 32,
    faiss_nprobe: int = 32,
    faiss_pq_subvectors: int = 8,
    faiss_pq_bits: int = 8,
    faiss_use_gpu: Any = "auto",
    faiss_gpu_device: int = 0,
) -> ad.AnnData:
    """Construct a genuine MetaSpot expression + within-cell spatial graph."""
    graph_started = _time.perf_counter()
    timing_seconds: Dict[str, float] = {}
    if adata1.n_obs == 0:
        raise ValueError("Cannot build a MetaSpot graph with zero observations")
    profile = str(execution_profile).strip().lower()
    if profile not in {"exact", "sampled", "streaming"}:
        raise ValueError(
            "execution_profile must be one of {'exact','sampled','streaming'}"
        )
    effective_knn_backend = str(knn_backend).strip().lower().replace(
        "-", "_"
    )
    if effective_knn_backend == "auto":
        effective_knn_backend = "exact"
    if (
        profile == "exact"
        and int(max_full_batch_nodes) > 0
        and adata1.n_obs > int(max_full_batch_nodes)
    ):
        raise ValueError(
            f"MetaSpot full-batch graph mode received {adata1.n_obs:,} nodes, "
            f"above module7.max_full_batch_nodes={int(max_full_batch_nodes):,}. "
            "Use execution.profile: sampled/streaming with a supported ANN "
            "backend; increasing this limit only removes the exact-path guard."
        )
    if float(alpha_expr) < 0 or float(alpha_spatial) < 0:
        raise ValueError("MetaSpot graph fusion weights must be non-negative")
    weight_sum = float(alpha_expr) + float(alpha_spatial)
    if weight_sum <= 0:
        raise ValueError("At least one MetaSpot graph fusion weight must be > 0")

    step_started = _time.perf_counter()
    adata1 = _attach_metaspot_spatial_metadata(adata1, point_df)
    timing_seconds["attach_spatial_metadata"] = (
        _time.perf_counter() - step_started
    )
    step_started = _time.perf_counter()
    expression = _metaspot_expression_embedding(
        adata1,
        n_components=expr_n_components,
    )
    timing_seconds["expression_embedding"] = _time.perf_counter() - step_started
    coordinates = adata1.obs[
        ["x_centroid", "y_centroid"]
    ].to_numpy(np.float32)
    from .scalable_graph import build_knn_connectivity

    step_started = _time.perf_counter()
    expression_graph, ann_metadata = build_knn_connectivity(
        expression,
        n_neighbors=k_expr,
        backend=effective_knn_backend,
        excluded_groups=(
            adata1.obs["cell_id"].astype(str).to_numpy()
            if expression_cross_cell_only
            else None
        ),
        candidate_multiplier=ann_candidate_multiplier,
        query_chunk_size=ann_query_chunk_nodes,
        train_sample=ann_train_sample,
        hnsw_m=faiss_hnsw_m,
        nprobe=faiss_nprobe,
        pq_subvectors=faiss_pq_subvectors,
        pq_bits=faiss_pq_bits,
        faiss_use_gpu=faiss_use_gpu,
        faiss_gpu_device=faiss_gpu_device,
        random_seed=42,
        n_jobs=_preferred_n_jobs(),
    )
    timing_seconds["expression_knn"] = _time.perf_counter() - step_started
    step_started = _time.perf_counter()
    spatial_graph = _within_cell_spatial_connectivity(
        coordinates,
        adata1.obs["cell_id"].astype(str).to_numpy(),
        k_spatial,
    )
    timing_seconds["within_cell_spatial"] = _time.perf_counter() - step_started
    if expression_cross_cell_only:
        if int(ann_metadata.get("nodes_below_available_degree", 0)) > 0:
            raise RuntimeError(
                "Cross-cell MetaSpot expression graph could not provide all "
                "globally available neighbors; increase ANN search quality "
                "or use module7.knn_backend='exact'"
            )
        if expression_graph.nnz == 0:
            raise ValueError(
                "Cross-cell MetaSpot expression graph contains no edges; "
                "disable module7.metaspot_expr_cross_cell_only or provide "
                "more than one cell"
            )

    step_started = _time.perf_counter()
    expr_weight = float(alpha_expr) / weight_sum
    spatial_weight = float(alpha_spatial) / weight_sum
    fused = (
        expr_weight * expression_graph
        + spatial_weight * spatial_graph
    ).tocsr()
    fused = ((fused + fused.T) * 0.5).tocsr()
    fused.eliminate_zeros()
    fused.sort_indices()

    adata1.obsm["X_expr_metaspot"] = expression
    adata1.obsm["X_expr_metaspot_scaled"] = expression.copy()
    feature_keys = ["X_expr_metaspot_scaled"]

    if use_geometry_features:
        geometry_columns = [
            column
            for column in [
                "r_mean",
                "r_std",
                "r_p50",
                "d_cell_norm_mean",
                "d_nuc_norm_mean",
                "elongation",
                "spread",
                "log_n",
            ]
            if column in adata1.obs.columns
        ]
        geometry = adata1.obs[geometry_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        for column_index in range(geometry.shape[1]):
            values = geometry[:, column_index]
            finite = np.isfinite(values)
            fill_value = float(np.median(values[finite])) if finite.any() else 0.0
            values[~finite] = fill_value
            geometry[:, column_index] = values
        geometry = StandardScaler().fit_transform(geometry).astype(np.float32)
        geometry[~np.isfinite(geometry)] = 0.0
        adata1.obsm["X_geom_metaspot_scaled"] = geometry
        feature_keys.append("X_geom_metaspot_scaled")

    adata1.obsp["metaspot_expr_connectivities"] = expression_graph
    adata1.obsp["metaspot_spatial_connectivities"] = spatial_graph
    adata1.obsp[METASPOT_GRAPH_KEY] = _normalize_sparse_adjacency(
        fused,
        mode=edge_norm,
    )
    checksum = _obs_names_checksum(adata1.obs_names)
    timing_seconds["fusion_and_attach"] = _time.perf_counter() - step_started
    timing_seconds["total"] = _time.perf_counter() - graph_started
    adata1.uns["metaspot_graph"] = {
        "schema_version": int(METASPOT_GRAPH_SCHEMA_VERSION),
        "node_unit": "metaspot",
        "node_id_checksum": checksum,
        "n_nodes": int(adata1.n_obs),
        "k_expr": int(min(max(0, k_expr), max(0, adata1.n_obs - 1))),
        "k_spatial": int(max(0, k_spatial)),
        "alpha_expr": expr_weight,
        "alpha_spatial": spatial_weight,
        "edge_norm": str(edge_norm).lower(),
        "expression_edges_nnz": int(expression_graph.nnz),
        "spatial_edges_nnz": int(spatial_graph.nnz),
        "fused_edges_nnz": int(fused.nnz),
        "spatial_edges_within_cell_only": True,
        "expression_edges_cross_cell_only": bool(
            expression_cross_cell_only
        ),
        "uses_boundary_geometry_features": bool(use_geometry_features),
        "max_full_batch_nodes": int(max_full_batch_nodes),
        "execution_profile": profile,
        "knn_backend": str(ann_metadata.get("backend", knn_backend)),
        "ann_metadata": ann_metadata,
        "timing_seconds": {
            key: float(value) for key, value in timing_seconds.items()
        },
        "feature_keys": feature_keys,
    }
    return adata1


def _metaspot_cache_matches(
    cached: ad.AnnData,
    reference: ad.AnnData,
    *,
    required_obsm: Sequence[str] = (),
    required_obsp: Sequence[str] = (),
) -> Tuple[bool, str]:
    """Validate MetaSpot graph/embedding caches before downstream reuse."""
    cached_names = pd.Index(cached.obs_names.astype(str))
    reference_names = pd.Index(reference.obs_names.astype(str))
    if not cached_names.equals(reference_names):
        return False, "MetaSpot observation order differs"
    for key in required_obsm:
        if key not in cached.obsm or cached.obsm[key].shape[0] != cached.n_obs:
            return False, f"missing or misaligned obsm[{key!r}]"
    for key in required_obsp:
        if key not in cached.obsp or cached.obsp[key].shape != (
            cached.n_obs,
            cached.n_obs,
        ):
            return False, f"missing or misaligned obsp[{key!r}]"
    expected_checksum = _obs_names_checksum(reference_names)
    graph_meta = cached.uns.get("metaspot_graph", {})
    if graph_meta and graph_meta.get("node_id_checksum") != expected_checksum:
        return False, "MetaSpot graph node checksum differs"
    return True, "aligned"


def _validate_adata1_point_df_alignment(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    geometry_atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Verify that a cached Module-4 object matches the current MetaSpots.

    MetaSpot identifiers are local cluster numbers (for example ``cell:17``)
    and can be reassigned when Module 3 is recomputed.  Set equality alone is
    therefore insufficient: the transcript counts and geometry attached to
    every identifier must also agree before a cached ``adata1`` can be reused.
    """
    if METASPOT_COL not in point_df.columns:
        return False, f"point_df is missing {METASPOT_COL!r}"
    if not adata1.obs_names.is_unique:
        return False, "cached adata1 has duplicate MetaSpot identifiers"
    if len(point_df) == 0:
        return (
            (adata1.n_obs == 0),
            "empty point_df" if adata1.n_obs == 0 else "point_df is empty",
        )

    metaspot_ids = point_df[METASPOT_COL].astype(str)
    point_counts = (
        point_df.groupby(metaspot_ids, observed=True, sort=False)
        .size()
        .astype(np.int64)
    )
    cached_ids = pd.Index(adata1.obs_names.astype(str))
    point_ids = pd.Index(point_counts.index.astype(str))
    missing = point_ids.difference(cached_ids)
    extra = cached_ids.difference(point_ids)
    if len(missing) or len(extra):
        return (
            False,
            "MetaSpot identifier mismatch "
            f"(missing_in_cache={len(missing)}, extra_in_cache={len(extra)})",
        )

    point_counts.index = point_ids
    expected_counts = point_counts.reindex(cached_ids).to_numpy(np.int64)
    cached_counts = np.asarray(adata1.X.sum(axis=1)).ravel()
    if cached_counts.shape != expected_counts.shape or not np.allclose(
        cached_counts,
        expected_counts,
        rtol=0.0,
        atol=0.0,
    ):
        mismatched = int(
            np.count_nonzero(
                ~np.isclose(
                    cached_counts,
                    expected_counts,
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
        return False, f"per-MetaSpot transcript counts differ ({mismatched} rows)"

    if "r_signed" in point_df.columns and "r_mean" in adata1.obs.columns:
        point_r_mean = (
            pd.to_numeric(point_df["r_signed"], errors="coerce")
            .groupby(metaspot_ids, observed=True, sort=False)
            .mean()
        )
        point_r_mean.index = point_r_mean.index.astype(str)
        expected_r = point_r_mean.reindex(cached_ids).to_numpy(dtype=float)
        cached_r = pd.to_numeric(
            adata1.obs["r_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite = np.isfinite(expected_r) & np.isfinite(cached_r)
        if not np.array_equal(np.isfinite(expected_r), np.isfinite(cached_r)):
            return False, "per-MetaSpot geometry missingness differs"
        if finite.any() and not np.allclose(
            cached_r[finite],
            expected_r[finite],
            rtol=0.0,
            atol=max(float(geometry_atol), 0.0),
        ):
            max_abs = float(
                np.max(np.abs(cached_r[finite] - expected_r[finite]))
            )
            return (
                False,
                "per-MetaSpot r_mean differs "
                f"(max_abs_difference={max_abs:.6g})",
            )

    return True, "aligned"


def _select_gc_feats_simple(adata1: ad.AnnData) -> List[str]:
    base_feats = [
        "r_mean", "r_std", "r_p10", "r_p50", "r_p90",
        "d_cell_norm_mean", "d_nuc_norm_mean",
        "elongation", "spread", "log_n",
    ]
    comp_feats = [c for c in adata1.obs.columns if c.startswith("comp_emb") and c.endswith("_mean")]
    comp_feats = comp_feats[: min(16, len(comp_feats))]
    if "local_density_mean" in adata1.obs.columns:
        base_feats.append("local_density_mean")
    return base_feats + comp_feats


def _spatial_feats_simple(adata1: ad.AnnData, point_df: pd.DataFrame) -> pd.DataFrame:
    metaspots = pd.Categorical(
        point_df[METASPOT_COL],
        categories=adata1.obs_names,
        ordered=False,
    )
    x = point_df["x_location"].to_numpy(np.float64, copy=False)
    y = point_df["y_location"].to_numpy(np.float64, copy=False)
    df = pd.DataFrame({METASPOT_COL: metaspots, "x": x, "y": y})
    agg = (
        df.groupby(METASPOT_COL, observed=True)
          .agg(
              n=("x", "size"),
              sx=("x", "sum"),
              sy=("y", "sum"),
              sxx=("x", lambda v: np.square(v).sum()),
              syy=("y", lambda v: np.square(v).sum()),
              sxy=("x", lambda v: (v * df.loc[v.index, "y"]).sum()),
          )
          .reindex(adata1.obs_names, fill_value=0)
    )
    n = agg["n"].to_numpy(np.float64)
    n_safe = np.maximum(n, 1)
    sx = agg["sx"].to_numpy(np.float64)
    sy = agg["sy"].to_numpy(np.float64)
    sxx = agg["sxx"].to_numpy(np.float64)
    syy = agg["syy"].to_numpy(np.float64)
    sxy = agg["sxy"].to_numpy(np.float64)
    mx, my = sx / n_safe, sy / n_safe
    Er2 = (sxx + syy) / n_safe
    Emu2 = mx * mx + my * my
    rms_radius = np.sqrt(np.maximum(Er2 - Emu2, 0.0))
    compactness = 1.0 / (rms_radius + 1.0)
    var_x = sxx / n_safe - mx * mx
    var_y = syy / n_safe - my * my
    cov_xy = sxy / n_safe - mx * my
    tr = var_x + var_y
    det = var_x * var_y - cov_xy * cov_xy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    sq = np.sqrt(disc)
    lam1 = 0.5 * (tr + sq)
    lam2 = 0.5 * (tr - sq)
    angle_consistency = lam1 / (lam2 + 1e-9)
    mask_small = n < 3
    compactness[mask_small] = 0.0
    angle_consistency[mask_small] = 0.0
    out = pd.DataFrame({
        "compactness": compactness.astype(np.float32),
        "angle_consistency": angle_consistency.astype(np.float32),
    }, index=adata1.obs_names)
    return out


def _construct_features_simple(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    speed_preset: str = "ultra",
    use_spatial_feats: bool = True,
    use_decell: bool = True,
    use_cosine: bool = True,
    w_gc: float = 0.4,
    w_expr: float = 0.6,
    w_sp: float = 0.0,
    use_gc: bool = True,
    use_expr: bool = True,
):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import normalize
    # 1) GC + composition
    feats_gc = _select_gc_feats_simple(adata1)
    X_gc = adata1.obs[feats_gc].to_numpy(np.float32, copy=False)
    X_gc = SimpleImputer(strategy="median").fit_transform(X_gc).astype(np.float32)
    X_gc = StandardScaler().fit_transform(X_gc).astype(np.float32)
    if not use_gc:
        X_gc = np.zeros_like(X_gc, dtype=np.float32)
    # 2) Expression TF-IDF + SVD
    X_counts = adata1.X.tocsr().astype(np.float32) if sp.issparse(adata1.X) else sp.csr_matrix(np.asarray(adata1.X, np.float32))
    row_sum = np.asarray(X_counts.sum(1)).ravel().astype(np.float32)
    row_sum[row_sum <= 0] = 1.0
    TF = X_counts.multiply(1.0 / row_sum[:, None])
    df_gene = np.asarray((X_counts > 0).sum(0)).ravel().astype(np.float32)
    N = X_counts.shape[0]
    idf = np.log1p(N / (1.0 + df_gene)).astype(np.float32)
    TFIDF = TF.multiply(idf)
    # Align expression SVD params with legacy defaults
    EXPR_NCOMP, SVD_N_ITER = 24, 3
    svd = TruncatedSVD(n_components=EXPR_NCOMP, random_state=42, n_iter=SVD_N_ITER)
    Z_expr = svd.fit_transform(TFIDF).astype(np.float32)
    Z_expr = StandardScaler().fit_transform(Z_expr).astype(np.float32)
    if not use_expr:
        Z_expr = np.zeros_like(Z_expr, dtype=np.float32)
    # 3) spatial shape
    if use_spatial_feats:
        spatial = _spatial_feats_simple(adata1, point_df)
        adata1.obs[["compactness", "angle_consistency"]] = spatial[["compactness", "angle_consistency"]]
        X_sp = adata1.obs[["compactness", "angle_consistency"]].fillna(0).to_numpy(np.float32)
        X_sp = StandardScaler().fit_transform(X_sp).astype(np.float32)
    else:
        X_sp = np.zeros((adata1.n_obs, 2), np.float32)
    # 4) combine with weights (configurable)
    X_base = np.hstack([float(w_gc) * X_gc, float(w_expr) * Z_expr, float(w_sp) * X_sp]).astype(np.float32)
    # 5) de-cell mean
    if "cell_id" not in adata1.obs.columns:
        tmp = (
            point_df[[METASPOT_COL, "cell_id"]]
            .drop_duplicates()
            .set_index(METASPOT_COL)
        )
        adata1.obs = adata1.obs.join(tmp, how="left")
    cell_ids = adata1.obs["cell_id"].astype(str).values
    u_cells, cell_codes = np.unique(cell_ids, return_inverse=True)
    if use_decell:
        d = X_base.shape[1]
        subs_per_cell = np.bincount(cell_codes)
        sum_by_cell = np.zeros((len(u_cells), d), np.float32)
        np.add.at(sum_by_cell, cell_codes, X_base)
        mean_by_cell = (sum_by_cell / np.maximum(1, subs_per_cell)[:, None]).astype(np.float32)
        X_all = X_base - mean_by_cell[cell_codes]
    else:
        X_all = X_base
    # 6) global standardize + L2
    X_all = StandardScaler().fit_transform(X_all).astype(np.float32)
    if use_cosine:
        X_all = normalize(X_all, norm="l2")
    X_all[~np.isfinite(X_all)] = 0.0
    return X_all, cell_ids, u_cells, cell_codes


def _choose_K_simple(
    n_sub: int,
    n_cells: int,
    target_sub_per_cluster: int = 800,
    target_cluster_per_cell: float = 1.0,
    K_min: int = 200,
    K_max: int = 8000,
):
    if n_sub <= 0:
        return max(2, K_min)
    K_by_size = max(50, int(n_sub / max(1, target_sub_per_cluster)))
    K_by_cells = int(np.clip(target_cluster_per_cell * n_cells, 50, K_max))
    K = min(K_by_size, K_by_cells)
    K = int(np.clip(K, K_min, min(K_max, n_sub // 10)))
    return max(2, K)


def _stratified_metaspot_sample(
    cell_ids: np.ndarray,
    sample_n: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample MetaSpots while preventing large cells from dominating."""
    cell_ids = np.asarray(cell_ids).astype(str)
    n = int(cell_ids.size)
    sample_n = max(1, min(int(sample_n), n))
    rng = np.random.default_rng(int(seed))
    if sample_n >= n:
        return rng.permutation(n).astype(np.int64, copy=False)

    _, cell_codes = np.unique(cell_ids, return_inverse=True)
    n_cells = int(cell_codes.max()) + 1 if cell_codes.size else 0
    if n_cells == 0:
        return rng.choice(n, size=sample_n, replace=False)

    # Sort once so looking up a cell does not repeatedly scan all MetaSpots.
    # This matters when both n and the number of cells are very large.
    order = np.argsort(cell_codes, kind="stable")
    cell_counts = np.bincount(cell_codes, minlength=n_cells)
    cell_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(cell_counts, dtype=np.int64),
        ]
    )

    def _members(code: int) -> np.ndarray:
        return order[cell_offsets[code]:cell_offsets[code + 1]]

    selected: List[np.ndarray] = []
    if sample_n < n_cells:
        chosen_cells = rng.choice(n_cells, size=sample_n, replace=False)
        for code in chosen_cells:
            members = _members(int(code))
            selected.append(
                np.asarray([rng.choice(members)], dtype=np.int64)
            )
    else:
        quota = max(1, sample_n // n_cells)
        for code in range(n_cells):
            members = _members(code)
            take = min(quota, members.size)
            if take:
                selected.append(
                    rng.choice(members, size=take, replace=False)
                )

    chosen = (
        np.concatenate(selected).astype(np.int64, copy=False)
        if selected
        else np.empty(0, dtype=np.int64)
    )
    chosen = np.unique(chosen)
    remaining = sample_n - chosen.size
    if remaining > 0:
        available = np.ones(n, dtype=bool)
        available[chosen] = False
        pool = np.flatnonzero(available)
        chosen = np.concatenate(
            [chosen, rng.choice(pool, size=remaining, replace=False)]
        )
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def _assign_to_metadomain_centers(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    use_cosine: bool,
    chunk_size: int,
) -> Tuple[np.ndarray, float]:
    """Assign rows to centers and return mean held-out distortion."""
    from sklearn.preprocessing import normalize

    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    n = int(values.shape[0])
    labels = np.empty(n, dtype=np.int32)
    total_loss = 0.0
    step = max(256, int(chunk_size))

    if use_cosine:
        centers_eval = normalize(centers, norm="l2").astype(
            np.float32,
            copy=False,
        )
        for start in range(0, n, step):
            end = min(n, start + step)
            similarity = values[start:end] @ centers_eval.T
            local_labels = np.argmax(similarity, axis=1)
            labels[start:end] = local_labels.astype(np.int32, copy=False)
            total_loss += float(
                np.sum(
                    1.0
                    - similarity[
                        np.arange(end - start),
                        local_labels,
                    ]
                )
            )
    else:
        center_norm = np.einsum("ij,ij->i", centers, centers)
        for start in range(0, n, step):
            end = min(n, start + step)
            chunk = values[start:end]
            chunk_norm = np.einsum("ij,ij->i", chunk, chunk)
            distance = (
                chunk_norm[:, None]
                + center_norm[None, :]
                - 2.0 * (chunk @ centers.T)
            )
            local_labels = np.argmin(distance, axis=1)
            labels[start:end] = local_labels.astype(np.int32, copy=False)
            total_loss += float(
                np.sum(
                    distance[
                        np.arange(end - start),
                        local_labels,
                    ]
                )
            )
    return labels, total_loss / max(1, n)


def _sampled_knee_candidates(
    *,
    n_sub: int,
    train_n: int,
    auto_k_min: int,
    auto_k_max: int,
    min_average_size: int,
    max_average_size: int,
    min_train_per_cluster: int,
    n_candidates: int,
) -> np.ndarray:
    """Build a logarithmic K grid with data-size and sample-size guards."""
    low = max(
        2,
        int(auto_k_min),
        int(math.ceil(n_sub / max(1, int(max_average_size)))),
    )
    high = min(
        int(auto_k_max),
        max(2, int(n_sub // max(1, int(min_average_size)))),
        max(2, int(train_n // max(1, int(min_train_per_cluster)))),
    )
    if high < low:
        low = high
    if high <= 2:
        return np.asarray([2], dtype=np.int32)
    if low == high:
        return np.asarray([low], dtype=np.int32)

    candidates = np.unique(
        np.rint(
            np.geomspace(
                low,
                high,
                num=max(3, int(n_candidates)),
            )
        ).astype(np.int32)
    )
    return candidates[(candidates >= 2) & (candidates <= high)]


def _select_sampled_knee(
    candidate_k: np.ndarray,
    distortion: np.ndarray,
    stability: np.ndarray,
    small_fraction: np.ndarray,
    *,
    stability_min: float,
    max_small_fraction: float,
    fallback_k: int,
) -> Tuple[int, int, str, np.ndarray]:
    """Select the distortion knee, backing off when quality guards fail."""
    candidate_k = np.asarray(candidate_k, dtype=np.int32)
    distortion = np.asarray(distortion, dtype=float)
    stability = np.asarray(stability, dtype=float)
    small_fraction = np.asarray(small_fraction, dtype=float)
    finite = np.isfinite(distortion)
    if not finite.any():
        chosen = int(
            candidate_k[
                np.argmin(np.abs(candidate_k.astype(float) - fallback_k))
            ]
        )
        return chosen, chosen, "sqrt_fallback_no_valid_fit", finite

    valid_indices = np.flatnonzero(finite)
    k_valid = candidate_k[valid_indices].astype(float)
    d_valid = distortion[valid_indices]
    if len(valid_indices) >= 3 and np.ptp(d_valid) > 1e-12:
        x = np.log(k_valid)
        x = (x - x.min()) / max(np.ptp(x), 1e-12)
        y = (d_valid - d_valid.min()) / max(np.ptp(d_valid), 1e-12)
        knee_local = int(np.argmax((1.0 - x) - y))
        knee_index = int(valid_indices[knee_local])
    else:
        knee_index = int(
            valid_indices[
                np.argmin(
                    np.abs(k_valid - float(fallback_k))
                )
            ]
        )
    knee_k = int(candidate_k[knee_index])

    stability_ok = np.isnan(stability) | (
        stability >= float(stability_min)
    )
    small_ok = np.isnan(small_fraction) | (
        small_fraction <= float(max_small_fraction)
    )
    feasible = finite & stability_ok & small_ok

    at_or_below_knee = np.flatnonzero(
        feasible & (candidate_k <= knee_k)
    )
    if at_or_below_knee.size:
        selected_index = int(at_or_below_knee[-1])
        reason = (
            "distortion_knee"
            if selected_index == knee_index
            else "guarded_lower_k"
        )
        return (
            int(candidate_k[selected_index]),
            knee_k,
            reason,
            feasible,
        )

    feasible_indices = np.flatnonzero(feasible)
    if feasible_indices.size:
        selected_index = int(
            feasible_indices[
                np.argmin(
                    np.abs(
                        candidate_k[feasible_indices].astype(float)
                        - float(knee_k)
                    )
                )
            ]
        )
        return (
            int(candidate_k[selected_index]),
            knee_k,
            "nearest_feasible_k",
            feasible,
        )

    selected_index = int(
        valid_indices[
            np.argmin(
                np.abs(k_valid - float(fallback_k))
            )
        ]
    )
    return (
        int(candidate_k[selected_index]),
        knee_k,
        "sqrt_fallback_no_feasible_k",
        feasible,
    )


def _choose_K_sampled_knee(
    values: np.ndarray,
    cell_ids: np.ndarray,
    *,
    use_cosine: bool,
    config: Mapping[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    """Choose MetaDomain K on a bounded sample using a held-out knee.

    Candidate models are fit only on a cell-balanced sample. The selected
    model size is the held-out cosine/Euclidean distortion knee, with
    cross-seed stability and tiny-cluster guards. This keeps automatic K
    selection bounded for very large datasets.
    """
    values = np.asarray(values, dtype=np.float32)
    n_sub = int(values.shape[0])
    if n_sub < 4:
        selected = max(2, min(n_sub, int(round(math.sqrt(n_sub)))))
        return selected, {
            "method": "sampled_knee",
            "selection_reason": "tiny_dataset_fallback",
            "selected_k": int(selected),
            "knee_k": int(selected),
            "candidate_k": np.asarray([selected], dtype=np.int32),
            "distortion": np.asarray([np.nan], dtype=np.float32),
            "stability_ari": np.asarray([np.nan], dtype=np.float32),
            "small_cluster_fraction": np.asarray(
                [np.nan],
                dtype=np.float32,
            ),
        }

    train_cap = max(20, int(config.get("auto_k_train_sample", 50000)))
    validation_cap = max(
        20,
        int(config.get("auto_k_validation_sample", 3000)),
    )
    validation_n = min(
        validation_cap,
        max(20, int(round(0.25 * n_sub))),
        n_sub - 2,
    )
    train_n = min(train_cap, n_sub - validation_n)
    if train_n < 20:
        train_n = max(2, n_sub - max(1, n_sub // 5))
        validation_n = n_sub - train_n

    seeds_raw = config.get("auto_k_seeds", [42, 43])
    if isinstance(seeds_raw, (int, np.integer)):
        seeds = [int(seeds_raw)]
    elif isinstance(seeds_raw, Sequence) and not isinstance(
        seeds_raw,
        (str, bytes),
    ):
        seeds = [int(seed) for seed in seeds_raw]
    else:
        seeds = [42, 43]
    seeds = list(dict.fromkeys(seeds)) or [42, 43]

    sample_seed = int(config.get("auto_k_sample_seed", 2026))
    sampled = _stratified_metaspot_sample(
        np.asarray(cell_ids),
        train_n + validation_n,
        seed=sample_seed,
    )
    train_indices = sampled[:train_n]
    validation_indices = sampled[train_n:]
    X_train = values[train_indices]
    X_validation = values[validation_indices]

    auto_k_max = int(
        config.get("auto_k_max", config.get("K_max", 8000))
    )
    candidates = _sampled_knee_candidates(
        n_sub=n_sub,
        train_n=train_n,
        auto_k_min=int(config.get("auto_k_min", 20)),
        auto_k_max=auto_k_max,
        min_average_size=int(
            config.get("auto_k_min_average_size", 20)
        ),
        max_average_size=int(
            config.get("auto_k_max_average_size", 500)
        ),
        min_train_per_cluster=int(
            config.get("auto_k_min_train_per_cluster", 10)
        ),
        n_candidates=int(config.get("auto_k_n_candidates", 8)),
    )

    search_batch_size = max(
        256,
        min(
            int(config.get("auto_k_batch_size", 2048)),
            train_n,
        ),
    )
    search_max_iter = max(
        5,
        int(config.get("auto_k_max_iter", 30)),
    )
    assign_chunk = max(
        256,
        int(config.get("assign_chunk_size", 50000)),
    )
    full_eval_max = max(
        validation_n,
        int(config.get("auto_k_full_eval_max", 100000)),
    )
    if n_sub <= full_eval_max:
        X_occupancy = values
        occupancy_scale = 1.0
    else:
        X_occupancy = values[sampled]
        occupancy_scale = n_sub / max(1, sampled.size)

    rows: List[Dict[str, float]] = []
    for candidate in candidates:
        validation_labels: List[np.ndarray] = []
        candidate_distortion: List[float] = []
        first_centers: Optional[np.ndarray] = None
        for seed in seeds:
            try:
                model = MiniBatchKMeans(
                    n_clusters=int(candidate),
                    init="k-means++",
                    n_init=1,
                    random_state=int(seed),
                    batch_size=search_batch_size,
                    max_iter=search_max_iter,
                    reassignment_ratio=0.01,
                    verbose=0,
                )
                model.fit(X_train)
                labels, loss = _assign_to_metadomain_centers(
                    X_validation,
                    model.cluster_centers_,
                    use_cosine=use_cosine,
                    chunk_size=assign_chunk,
                )
                validation_labels.append(labels)
                candidate_distortion.append(float(loss))
                if first_centers is None:
                    first_centers = model.cluster_centers_.astype(
                        np.float32,
                        copy=True,
                    )
            except Exception:
                validation_labels = []
                candidate_distortion = []
                first_centers = None
                break

        if not candidate_distortion or first_centers is None:
            rows.append(
                {
                    "k": float(candidate),
                    "distortion": np.nan,
                    "stability_ari": np.nan,
                    "small_cluster_fraction": np.nan,
                    "empty_cluster_fraction": np.nan,
                }
            )
            continue

        pairwise_stability: List[float] = []
        for left in range(len(validation_labels)):
            for right in range(left + 1, len(validation_labels)):
                pairwise_stability.append(
                    float(
                        adjusted_rand_score(
                            validation_labels[left],
                            validation_labels[right],
                        )
                    )
                )
        stability = (
            float(np.mean(pairwise_stability))
            if pairwise_stability
            else np.nan
        )

        occupancy_labels, _ = _assign_to_metadomain_centers(
            X_occupancy,
            first_centers,
            use_cosine=use_cosine,
            chunk_size=assign_chunk,
        )
        occupancy = np.bincount(
            occupancy_labels,
            minlength=int(candidate),
        ).astype(float)
        expected_full_occupancy = occupancy * occupancy_scale
        tiny = expected_full_occupancy < int(
            config.get("min_domain_size", 5)
        )
        small_fraction = float(
            occupancy[tiny].sum() / max(1, occupancy.sum())
        )
        rows.append(
            {
                "k": float(candidate),
                "distortion": float(np.mean(candidate_distortion)),
                "stability_ari": stability,
                "small_cluster_fraction": small_fraction,
                "empty_cluster_fraction": float(np.mean(occupancy == 0)),
            }
        )

    diagnostics = pd.DataFrame(rows).sort_values("k")
    candidate_array = diagnostics["k"].to_numpy(np.int32)
    fallback_k = int(
        np.clip(
            round(math.sqrt(n_sub)),
            candidate_array.min(),
            candidate_array.max(),
        )
    )
    stability_min = float(config.get("auto_k_stability_min", 0.4))
    max_small_fraction = float(
        config.get("auto_k_max_small_cluster_fraction", 0.01)
    )
    selected_k, knee_k, reason, feasible = _select_sampled_knee(
        candidate_array,
        diagnostics["distortion"].to_numpy(float),
        diagnostics["stability_ari"].to_numpy(float),
        diagnostics["small_cluster_fraction"].to_numpy(float),
        stability_min=stability_min,
        max_small_fraction=max_small_fraction,
        fallback_k=fallback_k,
    )

    result: Dict[str, Any] = {
        "method": "sampled_knee",
        "selection_reason": reason,
        "selected_k": int(selected_k),
        "knee_k": int(knee_k),
        "fallback_k": int(fallback_k),
        "train_size": int(train_n),
        "validation_size": int(validation_n),
        "use_cosine": bool(use_cosine),
        "stability_min": stability_min,
        "max_small_cluster_fraction": max_small_fraction,
        "candidate_k": candidate_array,
        "distortion": diagnostics["distortion"].to_numpy(np.float32),
        "stability_ari": diagnostics["stability_ari"].to_numpy(
            np.float32
        ),
        "small_cluster_fraction": diagnostics[
            "small_cluster_fraction"
        ].to_numpy(np.float32),
        "empty_cluster_fraction": diagnostics[
            "empty_cluster_fraction"
        ].to_numpy(np.float32),
        "feasible": np.asarray(feasible, dtype=bool),
    }
    return int(selected_k), result


def _cap_relax_factor(n_c: int) -> float:
    return float(1.0 + 0.6 * np.exp(-n_c / 80.0))


def _cap_per_cell_simple(
    labels: np.ndarray,
    X: np.ndarray,
    cell_ids: np.ndarray,
    cap_ratio_base: float = 0.015,
    min_keep: int = 3,
    margin_tau: float = 0.015,
    max_frac_soft: float = 0.05,
    use_cosine: bool = True,
):
    from scipy.spatial.distance import cdist
    y = labels.astype(np.int32, copy=True)
    n, d = X.shape
    if n == 0:
        return y
    K = int(y.max()) + 1
    centers = np.zeros((K, d), np.float32)
    for k in range(K):
        idx = (y == k).nonzero()[0]
        if idx.size:
            centers[k] = X[idx].mean(0)
    if use_cosine:
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        D = 1.0 - (Xn @ centers.T)
    else:
        D = cdist(X, centers, metric="euclidean")
    uniq, inv = np.unique(cell_ids.astype(str), return_inverse=True)
    for ci in range(len(uniq)):
        idx_c = (inv == ci).nonzero()[0]
        if idx_c.size == 0:
            continue
        y_c = y[idx_c]
        n_c = len(idx_c)
        relax = _cap_relax_factor(n_c)
        cap_abs = int(np.floor(cap_ratio_base * relax * n_c))
        cap_abs = max(min_keep, cap_abs)
        cap_abs = min(cap_abs, max(int(np.ceil(max_frac_soft * n_c)), min_keep))
        vals, cnts = np.unique(y_c, return_counts=True)
        for k, cnum in zip(vals, cnts):
            if cnum <= cap_abs:
                continue
            overflow = int(cnum - cap_abs)
            idx_ck = idx_c[(y_c == k).nonzero()[0]]
            d_cur = D[idx_ck, k]
            D_alt = D[idx_ck].copy(); D_alt[:, k] = np.inf
            alt_idx = np.argmin(D_alt, axis=1)
            d_alt = D[idx_ck, alt_idx]
            margin = (d_alt - d_cur) / np.maximum(d_cur, 1e-8)
            cand = np.where(margin < margin_tau)[0]
            if cand.size == 0:
                continue
            order = np.argsort(margin[cand])
            take = cand[order[:overflow]]
            y[idx_ck[take]] = alt_idx[take]
    return y


def _mini_batch_kmeans_simple(X: np.ndarray, K: int, seed: int = 0, desc: str = "MBK") -> np.ndarray:
    n = X.shape[0]
    if n <= K:
        return np.arange(n, dtype=np.int32)
    bs = int(max(1024, min(8192, int(n * (1/5)))))
    mbk = MiniBatchKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=1,
        random_state=seed,
        batch_size=bs,
        max_iter=60,
        reassignment_ratio=0.01,
        verbose=0,
    )
    y = mbk.fit_predict(X)
    return y.astype(np.int32)


def _two_stage_mb_kmeans_assign(
    X: np.ndarray,
    K: int,
    seed: int = 0,
    sample_n: int = 300000,
    batch_size: int = 8192,
    assign_chunk: int = 50000,
    use_cosine: bool = True,
    max_iter: int = 60,
):
    """Train MiniBatchKMeans on a subsample then assign all points in chunks.
    Keeps clustering quality close to full MBK while cutting runtime for very large blocks."""
    n = X.shape[0]
    if n <= K or n <= sample_n:
        return _mini_batch_kmeans_simple(X, K=K, seed=seed, desc="MBK-full")

    # 1) Train on a capped subsample (smaller default to reduce wall clock)
    sample_n = max(K, min(int(sample_n), n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_n, replace=False)
    X_train = X[idx]
    bs = max(1024, min(int(batch_size), sample_n))
    mbk = MiniBatchKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=3,
        random_state=seed,
        batch_size=bs,
        max_iter=int(max_iter),
        reassignment_ratio=0.01,
        verbose=0,
    )
    mbk.fit(X_train)
    centers = mbk.cluster_centers_.astype(np.float32, copy=False)

    # 2) Chunked assignment (fast BLAS dot for cosine, stabilized quadratic form for L2)
    if use_cosine:
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    step = max(2048, int(assign_chunk))
    labels = np.empty(n, dtype=np.int32)
    if use_cosine:
        for s in range(0, n, step):
            e = min(n, s + step)
            X_chunk = X[s:e]
            sim = X_chunk @ centers.T  # cosine similarity via dot product
            labels[s:e] = np.argmax(sim, axis=1).astype(np.int32)
    else:
        c_norm = np.einsum('ij,ij->i', centers, centers)
        for s in range(0, n, step):
            e = min(n, s + step)
            X_chunk = X[s:e]
            x_norm = np.einsum('ij,ij->i', X_chunk, X_chunk)
            # d^2 = ||x||^2 + ||c||^2 - 2 x·c
            dist = x_norm[:, None] + c_norm[None, :] - 2.0 * (X_chunk @ centers.T)
            labels[s:e] = np.argmin(dist, axis=1).astype(np.int32)
    return labels


def _metadomain_feature_matrix(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    cluster_rep: str,
    cluster_rep_decell: bool,
    speed_preset: str,
    use_spatial_feats: bool,
    use_decell: bool,
    use_cosine: bool,
    w_gc: float,
    w_expr: float,
    w_sp: float,
    use_gc: bool,
    use_expr: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select either the learned MetaSpot representation or legacy features."""
    from sklearn.preprocessing import normalize

    rep = str(cluster_rep).strip()
    if rep.lower() in {"", "legacy", "legacy_features", "auto"}:
        return _construct_features_simple(
            adata1,
            point_df,
            speed_preset=speed_preset,
            use_spatial_feats=use_spatial_feats,
            use_decell=use_decell,
            use_cosine=use_cosine,
            w_gc=w_gc,
            w_expr=w_expr,
            w_sp=w_sp,
            use_gc=use_gc,
            use_expr=use_expr,
        )

    if rep not in adata1.obsm:
        raise KeyError(
            f"module5.cluster_rep={rep!r} is missing from adata1.obsm; "
            "refusing to fall back to legacy features"
        )
    values = np.asarray(adata1.obsm[rep], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != adata1.n_obs:
        raise ValueError(
            f"adata1.obsm[{rep!r}] must have shape (n_metaspots, d), "
            f"got {values.shape}"
        )
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"adata1.obsm[{rep!r}] must be finite and have at least one feature"
        )

    if "cell_id" not in adata1.obs.columns:
        if not {METASPOT_COL, "cell_id"}.issubset(point_df.columns):
            raise ValueError(
                "Embedding-based MetaDomain clustering requires cell_id"
            )
        owners = (
            point_df[[METASPOT_COL, "cell_id"]]
            .assign(**{METASPOT_COL: lambda frame: frame[METASPOT_COL].astype(str)})
            .drop_duplicates()
            .set_index(METASPOT_COL)["cell_id"]
            .astype(str)
        )
        if owners.index.duplicated().any():
            raise ValueError("A MetaSpot has more than one owning cell")
        adata1.obs = adata1.obs.copy()
        adata1.obs["cell_id"] = owners.reindex(
            adata1.obs_names.astype(str)
        ).to_numpy()
    cell_ids = adata1.obs["cell_id"].astype(str).to_numpy()
    unique_cells, cell_codes = np.unique(cell_ids, return_inverse=True)

    matrix = StandardScaler().fit_transform(values).astype(np.float32)
    if cluster_rep_decell:
        sums = np.zeros((len(unique_cells), matrix.shape[1]), dtype=np.float32)
        np.add.at(sums, cell_codes, matrix)
        counts = np.bincount(cell_codes).astype(np.float32)
        means = sums / np.maximum(counts[:, None], 1.0)
        matrix = matrix - means[cell_codes]
        matrix = StandardScaler().fit_transform(matrix).astype(np.float32)
    if use_cosine:
        matrix = normalize(matrix, norm="l2").astype(np.float32)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix, cell_ids, unique_cells, cell_codes


def _merge_small_metadomain_clusters(
    labels: np.ndarray,
    values: np.ndarray,
    min_size: int,
    *,
    use_cosine: bool,
) -> np.ndarray:
    """Merge tiny clusters into the closest sufficiently sized prototype."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    min_size = max(1, int(min_size))
    if min_size <= 1 or labels.size == 0:
        return labels

    cluster_ids, counts = np.unique(labels, return_counts=True)
    large_ids = cluster_ids[counts >= min_size]
    small_ids = cluster_ids[counts < min_size]
    if large_ids.size == 0 or small_ids.size == 0:
        return labels

    centers = np.vstack(
        [values[labels == cluster_id].mean(axis=0) for cluster_id in large_ids]
    ).astype(np.float32)
    points = values
    if use_cosine:
        centers = centers / (
            np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
        )
        points = values / (
            np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
        )
        for cluster_id in small_ids:
            indices = np.flatnonzero(labels == cluster_id)
            nearest = np.argmax(points[indices] @ centers.T, axis=1)
            labels[indices] = large_ids[nearest]
    else:
        for cluster_id in small_ids:
            indices = np.flatnonzero(labels == cluster_id)
            nearest = pairwise_distances_argmin(values[indices], centers)
            labels[indices] = large_ids[nearest]
    return labels


def run_module5_simple(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    speed_preset: str = "ultra",
    use_spatial_feats: bool = True,
    use_decell: bool = True,
    use_cosine: bool = True,
    enable_cap: bool = True,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    module5_config_override: Optional[Mapping[str, Any]] = None,
):
    import time as _time
    _module_label = "Module 5 · Meta-domain Clustering"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 5 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # Load module5 config
    _m5 = {}
    try:
        _cfg = load_params_yaml()
        _m5 = _cfg.get('module5', {}) if _cfg else {}
    except Exception:
        _m5 = {}
    if module5_config_override:
        _m5 = {
            **_m5,
            **dict(module5_config_override),
        }
    w_gc = float(_m5.get('w_gc', 0.4))
    w_expr = float(_m5.get('w_expr', 0.6))
    w_sp = float(_m5.get('w_sp', 0.0))
    speed_preset = str(_m5.get('speed_preset', speed_preset))
    use_spatial_feats = bool(_m5.get('use_spatial_feats', use_spatial_feats))
    use_decell = bool(_m5.get('use_decell', use_decell))
    use_cosine = bool(_m5.get('use_cosine', use_cosine))
    enable_cap = bool(_m5.get('enable_cap', enable_cap))
    use_gc = bool(_m5.get('use_gc', True))
    use_expr = bool(_m5.get('use_expr', True))
    cluster_rep = str(_m5.get("cluster_rep", "legacy_features"))
    cluster_rep_decell = bool(_m5.get("cluster_rep_decell", False))
    clustering_method = str(
        _m5.get("clustering_method", "minibatch_kmeans")
    ).strip().lower().replace("-", "_")
    clustering_aliases = {
        "minibatch_kmeans": "minibatch_kmeans",
        "kmeans": "minibatch_kmeans",
        "legacy": "minibatch_kmeans",
        "consensus_leiden": "consensus_leiden",
        "stability_leiden": "consensus_leiden",
        "leiden": "consensus_leiden",
    }
    if clustering_method not in clustering_aliases:
        raise ValueError(
            "module5.clustering_method must be one of "
            "{'minibatch_kmeans', 'consensus_leiden'}"
        )
    clustering_method = clustering_aliases[clustering_method]
    large_scale_method = str(
        _m5.get("large_scale_method", "auto")
    ).strip().lower().replace("-", "_")
    if large_scale_method == "auto":
        effective_clustering_method = clustering_method
    elif large_scale_method in {
        "hierarchical_prototype",
        "hierarchical",
        "prototype",
    }:
        effective_clustering_method = "hierarchical_prototype"
    elif large_scale_method in {"consensus_leiden", "exact"}:
        effective_clustering_method = "consensus_leiden"
    elif large_scale_method == "minibatch_kmeans":
        effective_clustering_method = "minibatch_kmeans"
    else:
        raise ValueError(
            "module5.large_scale_method must be one of "
            "{'auto','minibatch_kmeans','consensus_leiden',"
            "'hierarchical_prototype'}"
        )
    min_domain_size = max(1, int(_m5.get("min_domain_size", 1)))
    random_seed = int(_m5.get("random_seed", 42))
    # More aggressive speed defaults for large cohorts; can still be overridden in params.yaml
    train_sample_per_block = int(_m5.get('train_sample_per_block', 300000))
    assign_chunk_size = int(_m5.get('assign_chunk_size', _m5.get('block_size', 50000) or 50000))
    mbk_batch_size = int(_m5.get('mbk_batch_size', 8192))
    mbk_max_iter = int(_m5.get('mbk_max_iter', 60 if str(speed_preset).lower() == 'ultra' else 80))
    # feature construction
    X_all, cell_ids, u_cells, cell_codes = _metadomain_feature_matrix(
        adata1,
        point_df,
        cluster_rep=cluster_rep,
        cluster_rep_decell=cluster_rep_decell,
        speed_preset=speed_preset,
        use_spatial_feats=use_spatial_feats,
        use_decell=use_decell,
        use_cosine=use_cosine,
        w_gc=w_gc,
        w_expr=w_expr,
        w_sp=w_sp,
        use_gc=use_gc,
        use_expr=use_expr,
    )
    n_sub = X_all.shape[0]
    n_cells = len(u_cells)
    if effective_clustering_method == "hierarchical_prototype":
        from .hierarchical_metadomain import (
            run_hierarchical_prototype_clustering,
        )

        split_by_nucleus = bool(_m5.get("split_by_nucleus", True))
        if split_by_nucleus and "r_p50" in adata1.obs.columns:
            threshold = float(_m5.get("nucleus_threshold", 0.0))
            side_labels = np.where(
                adata1.obs["r_p50"].to_numpy(np.float32) < threshold,
                "N",
                "C",
            )
        else:
            side_labels = np.full(n_sub, "A", dtype=object)
        t0_hierarchical = _step_start(
            "cell-aligned local prototypes and global prototype clustering"
        )
        global_labels, metadata, diagnostic_table = (
            run_hierarchical_prototype_clustering(
                X_all,
                cell_ids,
                side_labels=side_labels,
                config=_m5,
            )
        )
        global_labels = _merge_small_metadomain_clusters(
            global_labels,
            X_all,
            min_domain_size,
            use_cosine=use_cosine,
        )
        unique = np.unique(global_labels)
        remap = {int(value): index for index, value in enumerate(unique)}
        dense = np.asarray(
            [remap[int(value)] for value in global_labels],
            dtype=np.int32,
        )
        _step_done("hierarchical prototype clustering", t0_hierarchical)
        adata1.obs["meta-domain"] = pd.Categorical(dense.astype(str))
        metadata = {
            **metadata,
            "cluster_rep": cluster_rep,
            "cluster_rep_decell": bool(cluster_rep_decell),
            "split_by_nucleus": bool(split_by_nucleus),
            "min_domain_size": int(min_domain_size),
            "random_seed": int(random_seed),
            "n_metadomains_after_small_cluster_merge": int(unique.size),
        }
        adata1.uns["module5_metadomain_clustering"] = metadata
        _logging.getLogger("cellscope").info(
            "Module 5 hierarchical prototypes: MetaSpots=%d "
            "local_prototypes=%d MetaDomains=%d blocks=%d",
            int(n_sub),
            int(metadata["n_local_prototypes"]),
            int(unique.size),
            int(metadata["n_blocks"]),
        )
        return adata1, diagnostic_table

    if effective_clustering_method == "consensus_leiden":
        from .consensus_clustering import run_stability_consensus_leiden

        split_by_nucleus = bool(_m5.get("split_by_nucleus", True))
        if split_by_nucleus and "r_p50" in adata1.obs.columns:
            threshold = float(_m5.get("nucleus_threshold", 0.0))
            is_nuclear = (
                adata1.obs["r_p50"].to_numpy(np.float32) < threshold
            )
            blocks = [
                ("N", np.flatnonzero(is_nuclear)),
                ("C", np.flatnonzero(~is_nuclear)),
            ]
        else:
            blocks = [("A", np.arange(n_sub, dtype=np.int64))]
        global_labels = np.empty(n_sub, dtype=np.int32)
        offset = 0
        consensus_by_block: Dict[str, Any] = {}
        diagnostic_rows: List[Dict[str, Any]] = []
        t0_consensus = _step_start("stability consensus Leiden")
        for prefix, indices in blocks:
            if indices.size == 0:
                continue
            local_labels, metadata = run_stability_consensus_leiden(
                X_all[indices],
                cell_ids[indices],
                config=_m5,
                n_jobs=_preferred_n_jobs(),
            )
            local_labels = _merge_small_metadomain_clusters(
                local_labels,
                X_all[indices],
                min_domain_size,
                use_cosine=use_cosine,
            )
            unique = np.unique(local_labels)
            remap = {value: index for index, value in enumerate(unique)}
            dense = np.asarray(
                [remap[value] for value in local_labels],
                dtype=np.int32,
            )
            global_labels[indices] = dense + offset
            offset += int(dense.max()) + 1
            consensus_by_block[prefix] = metadata
            candidate_resolution = np.asarray(
                metadata["candidate_resolution"], dtype=float
            )
            selected_resolution = float(metadata["selected_resolution"])
            selected_index = int(
                np.argmin(np.abs(candidate_resolution - selected_resolution))
            )
            diagnostic_rows.append(
                {
                    "block": prefix,
                    "n_metaspots": int(indices.size),
                    "selected_resolution": selected_resolution,
                    "selected_stability_ari": float(
                        np.asarray(
                            metadata["candidate_stability_ari"], dtype=float
                        )[selected_index]
                    ),
                    "selected_candidate_clusters": int(
                        np.asarray(
                            metadata["candidate_n_clusters"], dtype=int
                        )[selected_index]
                    ),
                    "final_clusters": int(unique.size),
                    "selection_reason": str(metadata["selection_reason"]),
                }
            )
        _step_done("stability consensus Leiden", t0_consensus)
        adata1.obs["meta-domain"] = pd.Categorical(
            global_labels.astype(str)
        )
        adata1.uns["module5_metadomain_clustering"] = {
            "clustering_method": "consensus_leiden",
            "cluster_rep": cluster_rep,
            "cluster_rep_decell": bool(cluster_rep_decell),
            "split_by_nucleus": bool(split_by_nucleus),
            "min_domain_size": int(min_domain_size),
            "random_seed": int(random_seed),
            "n_metaspots": int(n_sub),
            "n_metadomains": int(
                adata1.obs["meta-domain"].nunique()
            ),
            "consensus_by_block": consensus_by_block,
        }
        _logging.getLogger("cellscope").info(
            "Module 5 consensus Leiden: MetaSpots=%d MetaDomains=%d blocks=%s",
            int(n_sub),
            int(adata1.obs["meta-domain"].nunique()),
            [row["block"] for row in diagnostic_rows],
        )
        return adata1, pd.DataFrame(diagnostic_rows)

    # K estimation. "auto" is a sampled held-out distortion knee; the
    # historical count-only heuristic remains available as "legacy_auto".
    estimate_k_mode = str(_m5.get("estimate_k", "auto")).strip().lower()
    if estimate_k_mode == "fixed":
        K_total = int(_m5.get('fixed_k', 1024))
        k_selection: Dict[str, Any] = {
            "method": "fixed",
            "selection_reason": "configured_fixed_k",
            "selected_k": int(K_total),
            "knee_k": int(K_total),
            "candidate_k": np.asarray([K_total], dtype=np.int32),
        }
    elif estimate_k_mode in {"legacy_auto", "legacy"}:
        K_total = _choose_K_simple(
            n_sub=n_sub,
            n_cells=n_cells,
            target_sub_per_cluster=int(_m5.get('target_sub_per_cluster', 800)),
            target_cluster_per_cell=float(_m5.get('target_cluster_per_cell', 1.0)),
            K_min=int(_m5.get('K_min', 200)),
            K_max=int(_m5.get('K_max', 8000)),
        )
        k_selection = {
            "method": "legacy_auto",
            "selection_reason": "legacy_count_heuristic",
            "selected_k": int(K_total),
            "knee_k": int(K_total),
            "candidate_k": np.asarray([K_total], dtype=np.int32),
        }
    elif estimate_k_mode in {"auto", "sampled_knee"}:
        t0_auto_k = _step_start("sampled automatic K selection")
        K_total, k_selection = _choose_K_sampled_knee(
            X_all,
            cell_ids,
            use_cosine=use_cosine,
            config=_m5,
        )
        _step_done("sampled automatic K selection", t0_auto_k)
        _logging.getLogger("cellscope").info(
            "Module 5 automatic K: selected=%d knee=%d reason=%s "
            "candidates=%s",
            int(K_total),
            int(k_selection.get("knee_k", K_total)),
            str(k_selection.get("selection_reason", "")),
            np.asarray(
                k_selection.get("candidate_k", [K_total])
            ).tolist(),
        )
    else:
        raise ValueError(
            "module5.estimate_k must be one of "
            "{'auto', 'sampled_knee', 'legacy_auto', 'fixed'}, "
            f"got {estimate_k_mode!r}"
        )
    # split by nucleus/cytoplasm using r_p50
    split_by_nucleus = bool(_m5.get('split_by_nucleus', True))
    if split_by_nucleus and ("r_p50" in adata1.obs.columns):
        thr = float(_m5.get('nucleus_threshold', 0.0))
        is_nu = adata1.obs["r_p50"].to_numpy(np.float32) < thr
        blocks = [("N", np.where(is_nu)[0]), ("C", np.where(~is_nu)[0])]
    else:
        blocks = [("A", np.arange(n_sub, dtype=int))]
    global_labels = np.empty(n_sub, dtype=np.int32)
    offset = 0
    # Announce clustering start (single callback update)
    t0_main = _step_start('block clustering')
    rng_blocks = _progress_iter(blocks, desc='block clustering', total=len(blocks), show=show_internal_progress)
    for prefix, idx in rng_blocks:
        if idx.size == 0:
            continue
        Xi = X_all[idx]
        cell_ids_blk = cell_ids[idx]
        n_cells_blk = np.unique(cell_ids_blk).size
        K_blk = max(2, int(round(K_total * (n_cells_blk / max(1, n_cells)))))
        y = _two_stage_mb_kmeans_assign(
            Xi,
            K=K_blk,
            seed=random_seed + (1 if prefix == 'C' else 0),
            sample_n=train_sample_per_block,
            batch_size=mbk_batch_size,
            assign_chunk=assign_chunk_size,
            use_cosine=use_cosine,
            max_iter=mbk_max_iter,
        )
        if enable_cap:
            y = _cap_per_cell_simple(
                y,
                Xi,
                cell_ids_blk,
                cap_ratio_base=float(_m5.get('cap_ratio_base', 0.015)),
                min_keep=int(_m5.get('cap_min_keep', 3)),
                margin_tau=float(_m5.get('cap_margin_tau', 0.015)),
                max_frac_soft=float(_m5.get('cap_max_frac_soft', 0.05)),
                use_cosine=use_cosine,
            )
        y = _merge_small_metadomain_clusters(
            y,
            Xi,
            min_domain_size,
            use_cosine=use_cosine,
        )
        u = np.unique(y); remap = {u_i: i for i, u_i in enumerate(u)}
        dense = np.vectorize(remap.get, otypes=[np.int32])(y)
        global_labels[idx] = dense + offset
        offset += dense.max() + 1
        del Xi; gc.collect()
    # Avoid duplicate print of final summary when internal iterator already printed timings.
    if not show_internal_progress:
        _step_done('block clustering', t0_main)
    adata1.obs["meta-domain"] = pd.Categorical(global_labels.astype(str))
    adata1.uns["module5_metadomain_clustering"] = {
        "clustering_method": "minibatch_kmeans",
        "cluster_rep": cluster_rep,
        "cluster_rep_decell": bool(cluster_rep_decell),
        "split_by_nucleus": bool(split_by_nucleus),
        "min_domain_size": int(min_domain_size),
        "random_seed": int(random_seed),
        "n_metaspots": int(n_sub),
        "n_metadomains": int(adata1.obs["meta-domain"].nunique()),
        "k_selection": k_selection,
    }
    return adata1, pd.DataFrame(
        {
            "n_sub": [n_sub],
            "n_cells": [n_cells],
            "K_total": [K_total],
            "knee_K": [int(k_selection.get("knee_k", K_total))],
            "selection_method": [
                str(k_selection.get("method", estimate_k_mode))
            ],
            "selection_reason": [
                str(k_selection.get("selection_reason", ""))
            ],
            "final_K": [
                int(adata1.obs["meta-domain"].nunique())
            ],
        }
    )


def build_adata2_from_adata1_and_pointdf(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    module6_config_override: Optional[Mapping[str, Any]] = None,
    graph_unit_override: Optional[str] = None,
    pool_output_key_map: Optional[Mapping[str, str]] = None,
) -> ad.AnnData:
    import time as _time
    _module_label = "Module 6 · Aggregate Domains into Cluster Feature Table (adata2)"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 6 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # 1) normalize rows (TF/CPM-like)
    # Optional overall progress bar for M6 (fixed 5 steps)
    _m6_prog = None; _m6_task = None
    if show_internal_progress:
        try:
            from rich.console import Console as _Console
            from rich.progress import Progress as _Progress, SpinnerColumn as _Spin, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time
            _m6_console = _Console(stderr=True)
            if _m6_console.is_terminal:
                _m6_prog = _Progress(_Spin(), _Text(" Building adata2"), _Bar(), _Time(), transient=True, console=_m6_console)
                _m6_task = _m6_prog.add_task("adata2", total=5)
                _m6_prog.start()
        except Exception:
            _m6_prog = None
    t0 = _step_start("Normalize counts (TF)")
    X1 = adata1.X.tocsr().astype(np.float32) if sp.issparse(adata1.X) else sp.csr_matrix(np.asarray(adata1.X, np.float32))
    rs = np.array(X1.sum(1)).ravel().astype(np.float32); rs[rs == 0] = 1.0
    X1_norm = X1.multiply(1.0 / rs[:, None])
    _step_done("Normalize counts (TF)", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 2) MetaSpot -> meta-domain mapping
    t0 = _step_start("Map metaspots to clusters")
    sub2clu = adata1.obs['meta-domain'].astype(str).values
    clu_codes, clu_uni = pd.factorize(sub2clu, sort=False)
    _step_done("Map metaspots to clusters", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 3) group mean to clusters.  Keep the true aggregate counts separate from
    # the compositional mean used by the downstream enrichment model.  SCVI
    # requires non-negative count-like input; feeding X2 (row-normalized
    # proportions) to a negative-binomial model is statistically invalid.
    t0 = _step_start("Aggregate metaspots -> cluster means")
    ones = np.ones_like(clu_codes, dtype=np.float32)
    G = sp.csr_matrix((ones, (clu_codes, np.arange(len(clu_codes)))), shape=(len(clu_uni), len(clu_codes)))
    row_sum = np.array(G.sum(1)).ravel(); row_sum[row_sum == 0] = 1.0
    X2 = (G.multiply(1.0 / row_sum[:, None]) @ X1_norm).tocsr()
    X2_counts = (G @ X1).tocsr().astype(np.float32, copy=False)
    adata2 = ad.AnnData(X=X2, obs=pd.DataFrame(index=clu_uni), var=adata1.var.copy())
    adata2.layers["counts"] = X2_counts
    adata2.obs["n_metaspots"] = row_sum.astype(np.int64)
    adata2.uns["matrix_semantics"] = {
        "X": "mean MetaSpot transcript-frequency profile; rows sum to approximately one",
        "layers_counts": "sum of raw MetaSpot gene counts per MetaDomain",
    }

    try:
        module6_config = (load_params_yaml() or {}).get("module6", {}) or {}
    except Exception:
        module6_config = {}
    if module6_config_override:
        module6_config = {
            **module6_config,
            **dict(module6_config_override),
        }
    pool_reps = module6_config.get("pool_reps", [])
    if isinstance(pool_reps, str):
        pool_reps = [
            value.strip()
            for value in pool_reps.split(",")
            if value.strip()
        ]
    elif not isinstance(pool_reps, (list, tuple)):
        pool_reps = []
    pool_normalize = bool(module6_config.get("pool_normalize", True))
    if graph_unit_override is None:
        graph_unit = str(
            ((load_params_yaml() or {}).get("module7", {}) or {}).get(
                "graph_unit",
                "metadomain",
            )
        ).strip().lower()
    else:
        graph_unit = str(graph_unit_override).strip().lower()

    pooled_metadata: Dict[str, Any] = {}
    group_mean = G.multiply(1.0 / row_sum[:, None]).tocsr()
    output_key_map = {
        METASPOT_DGI_KEY: METADOMAIN_DGI_KEY,
        METASPOT_DGI_SAGE_KEY: "X_dgi_sage_metadomain",
        "X_expr_metaspot": "X_expr_metadomain",
    }
    if pool_output_key_map:
        output_key_map.update(
            {
                str(source): str(target)
                for source, target in pool_output_key_map.items()
            }
        )
    compatibility_aliases = {
        METASPOT_DGI_KEY: "X_dgi_gcn",
        METASPOT_DGI_SAGE_KEY: "X_dgi_sage",
        "X_expr_metaspot": "X_expr",
    }
    for source_key in [str(value) for value in pool_reps]:
        if source_key not in adata1.obsm:
            if graph_unit == "metaspot":
                raise KeyError(
                    f"module6.pool_reps requested {source_key!r}, but it is "
                    "missing from adata1.obsm"
                )
            continue
        source = np.asarray(adata1.obsm[source_key], dtype=np.float32)
        if source.ndim != 2 or source.shape[0] != adata1.n_obs:
            raise ValueError(
                f"adata1.obsm[{source_key!r}] is not aligned to MetaSpots"
            )
        if not np.isfinite(source).all():
            raise ValueError(
                f"adata1.obsm[{source_key!r}] contains non-finite values"
            )
        if pool_normalize:
            source_norm = np.linalg.norm(source, axis=1, keepdims=True)
            source = source / np.maximum(source_norm, 1e-8)
        pooled = np.asarray(group_mean @ source, dtype=np.float32)
        if pool_normalize:
            pooled_norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(pooled_norm, 1e-8)
        output_key = output_key_map.get(
            source_key,
            f"{source_key}_metadomain",
        )
        adata2.obsm[output_key] = pooled
        alias = compatibility_aliases.get(source_key)
        if alias:
            adata2.obsm[alias] = pooled.copy()
        pooled_metadata[output_key] = {
            "source_key": source_key,
            "source_unit": "metaspot",
            "target_unit": "metadomain",
            "pooling": "equal_metaspot_mean",
            "l2_normalized": bool(pool_normalize),
            "compatibility_alias": alias or "",
        }
    if graph_unit == "metaspot" and METADOMAIN_DGI_KEY not in adata2.obsm:
        raise RuntimeError(
            "MetaSpot graph mode requires pooled MetaDomain DGI embedding "
            f"{METADOMAIN_DGI_KEY!r}"
        )
    if pooled_metadata:
        adata2.uns["embedding_pooling"] = {
            "schema_version": 1,
            "membership_checksum": _membership_checksum(
                adata1.obs_names,
                adata1.obs["meta-domain"].astype(str).to_numpy(),
            ),
            "representations": pooled_metadata,
        }
    _step_done("Aggregate metaspots -> cluster means", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 4) aggregate geometry stats
    t0 = _step_start("Aggregate geometry stats per cluster")
    geometry_columns = [
        column
        for column in (
            "r_mean",
            "r_std",
            "r_p50",
            "d_cell_norm_mean",
            "d_nuc_norm_mean",
            "elongation",
            "spread",
            "log_n",
        )
        if column in adata1.obs.columns
    ]
    if geometry_columns:
        aggregation = {
            column: (
                "median"
                if column in {"elongation", "spread"}
                else "mean"
            )
            for column in geometry_columns
        }
        agg = (
            adata1.obs[geometry_columns + ["meta-domain"]]
            .groupby("meta-domain", sort=False, observed=True)
            .agg(aggregation)
        )
        adata2.obs = adata2.obs.join(agg, how="left")
    _step_done("Aggregate geometry stats per cluster", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 5) representative cell id
    t0 = _step_start("Select representative cell per cluster")
    rep_cell = (adata1.obs[['meta-domain','cell_id']]
              .groupby('meta-domain', observed=True)['cell_id']
              .agg(lambda x: x.value_counts().index[0]))
    adata2.obs['rep_cell_id'] = adata2.obs.index.map(rep_cell.to_dict())
    _step_done("Select representative cell per cluster", t0)
    if _m6_prog and _m6_task is not None:
        try:
            _m6_prog.advance(_m6_task, 1)
            _m6_prog.stop()
        except Exception:
            pass
    return adata2


def _is_count_like_matrix(matrix, tolerance: float = 1e-4) -> bool:
    """Return True when a dense/sparse matrix is valid count-model input."""
    if matrix is None:
        return False
    X = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)
    values = X.data if sp.issparse(X) else X.ravel()
    if values.size == 0:
        return False
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -float(tolerance)):
        return False
    return bool(
        np.max(np.abs(values - np.rint(values)), initial=0.0)
        <= float(tolerance)
    )


def build_module7_graph(
    adata2: ad.AnnData,
    show_internal_progress: bool = False,
    scvi_epochs: int = 100,
    out_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ad.AnnData:
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import kneighbors_graph
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse import diags
    import time as _time

    # Parameters (read from config when available)
    GEOM_FEATS = ['r_mean','r_std','r_p50','d_cell_norm_mean','d_nuc_norm_mean','elongation','spread','log_n']
    EDGE_NORM = 'rw'
    k_expr_base, k_geom_base = 32, 32
    # Fusion weights
    W_EXPR_GRAPH = 0.5
    W_GEOM_GRAPH = 0.5
    TOPK = 32
    BETA_SAMECELL = 0.5
    scvi_n_latent = 32
    scvi_lr = 1e-3
    scvi_batch_size = 512
    feature_scale_mode = 'standard'

    try:
        _cfg = load_params_yaml()
        _m7 = _cfg.get('module7', {}) if _cfg else {}
        scvi_epochs = int(_m7.get('scvi_epochs', scvi_epochs))
        # neighbors
        _nei = _m7.get('neighbors', None)
        if isinstance(_nei, dict):
            k_expr_base = int(_nei.get('k_expr', k_expr_base))
            k_geom_base = int(_nei.get('k_geom', k_geom_base))
        elif isinstance(_nei, (int, float)):
            k_expr_base = k_geom_base = int(_nei)
        # weights and prune
        W_EXPR_GRAPH = float(_m7.get('alpha_expr', W_EXPR_GRAPH))
        W_GEOM_GRAPH = float(_m7.get('alpha_geom', W_GEOM_GRAPH))
        TOPK = int(_m7.get('topk_prune', TOPK))
        BETA_SAMECELL = float(_m7.get('same_cell_beta', BETA_SAMECELL))
        EDGE_NORM = str(_m7.get('edge_norm', EDGE_NORM)).lower()
        feature_scale_mode = str(_m7.get('feature_scale', feature_scale_mode)).lower()
        scvi_n_latent = int(_m7.get('scvi_n_latent', scvi_n_latent))
        scvi_lr = float(_m7.get('scvi_lr', scvi_lr))
        scvi_batch_size = max(1, int(_m7.get('scvi_batch_size', scvi_batch_size)))
    except Exception:
        pass

    def compute_expr_latent_scvi_or_svd(
        adata2: ad.AnnData,
        n_latent: int = 32,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        # If epochs <= 0, force SVD path (skip SCVI entirely)
        if int(max_epochs) <= 0:
            X = adata2.X.tocsr().astype(np.float32) if sp.issparse(adata2.X) else sp.csr_matrix(np.asarray(adata2.X, np.float32))
            row_sum = np.array(X.sum(1)).ravel().astype(np.float32); row_sum[row_sum == 0] = 1.0
            TF = X.multiply(1.0 / row_sum[:, None])
            df = np.array((X > 0).sum(0)).ravel().astype(np.float32)
            N = X.shape[0]; idf = np.log1p(N / (1.0 + df)).astype(np.float32)
            TFIDF = TF.multiply(idf)
            svd = TruncatedSVD(n_components=n_latent, random_state=42)
            Z = svd.fit_transform(TFIDF).astype(np.float32)
            return Z, "SVD", {"n_components": int(n_latent), "tfidf": True}
        try:
            import scvi
            count_layer = adata2.layers.get("counts", None)
            if not _is_count_like_matrix(count_layer):
                raise ValueError(
                    "SCVI requires a non-negative integer-like counts layer; "
                    "falling back to TF-IDF/SVD."
                )
            # Quiet mode: suppress progress bars and logs
            try:
                scvi.settings.verbosity = 0
            except Exception:
                pass
            # Silence Lightning banners (again, in case import order differs)
            try:
                import logging as _logging
                for _name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
                    _logging.getLogger(_name).setLevel(_logging.ERROR)
            except Exception:
                pass
            import warnings as _warnings
            with _warnings.catch_warnings():
                scvi.model.SCVI.setup_anndata(adata2, layer="counts")
            model = scvi.model.SCVI(adata2, n_latent=n_latent)
            import warnings as _warnings
            with _warnings.catch_warnings():
                model.train(
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    plan_kwargs={"lr": learning_rate},
                    check_val_every_n_epoch=None,
                    enable_progress_bar=False,
                    logger=False,
                    enable_model_summary=False,
                    log_every_n_steps=0,
                )
            Z = model.get_latent_representation().astype(np.float32)
            return Z, "SCVI", {
                "n_latent": int(n_latent),
                "epochs": int(max_epochs),
                "lr": float(learning_rate),
                "batch_size": int(batch_size),
            }
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "SCVI expression embedding unavailable or invalid; using "
                "TF-IDF/SVD instead: %s",
                exc,
            )
            X = adata2.X.tocsr().astype(np.float32) if sp.issparse(adata2.X) else sp.csr_matrix(np.asarray(adata2.X, np.float32))
            row_sum = np.array(X.sum(1)).ravel().astype(np.float32); row_sum[row_sum == 0] = 1.0
            TF = X.multiply(1.0 / row_sum[:, None])
            df = np.array((X > 0).sum(0)).ravel().astype(np.float32)
            N = X.shape[0]; idf = np.log1p(N / (1.0 + df)).astype(np.float32)
            TFIDF = TF.multiply(idf)
            svd = TruncatedSVD(n_components=n_latent, random_state=42)
            Z = svd.fit_transform(TFIDF).astype(np.float32)
            return Z, "SVD(fallback)", {"n_components": int(n_latent), "tfidf": True}

    def _build_knn_connectivities_by_scanpy(adata: ad.AnnData, rep_key: str, n_neighbors: int = 20) -> sp.csr_matrix:
            # Prefer sklearn kneighbors_graph for performance; fallback to scanpy
            try:
                G_ex = kneighbors_graph(adata.obsm[rep_key], n_neighbors=n_neighbors, mode='connectivity', include_self=False, n_jobs=_preferred_n_jobs())
                A = G_ex.maximum(G_ex.T).astype(np.float32).tocsr()
                A.eliminate_zeros(); A.sort_indices()
                return A
            except Exception as e:
                _logging.getLogger("cellscope").warning(f"kneighbors_graph failed: {e}; falling back to scanpy.pp.neighbors.")
                try:
                    import scanpy as sc
                    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep_key, method='umap', key_added=f'{rep_key}_nbrs')
                    C = adata.obsp.get(f'{rep_key}_nbrs_connectivities', None)
                    if C is None:
                        C = adata.obsp['connectivities']
                    C = C.tocsr().astype(np.float32)
                    C = ((C + C.T) * 0.5).tocsr()
                    C.eliminate_zeros(); C.sort_indices()
                    return C
                except Exception as e2:
                    _logging.getLogger("cellscope").error(f"scanpy neighbors failed: {e2}; returning empty CSR.")
                    n = adata.n_obs
                    return sp.csr_matrix((n, n), dtype=np.float32)

    def _row_topk_csr(A: sp.csr_matrix, k: int = 16) -> sp.csr_matrix:
        A = A.tocsr().astype(np.float32, copy=False)
        n = A.shape[0]; indptr, indices, data = A.indptr, A.indices, A.data
        rows, cols, vals = [], [], []
        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            if s == e:
                rows.append(np.array([i], dtype=np.int32)); cols.append(np.array([i], dtype=np.int32)); vals.append(np.array([1e-6], dtype=np.float32)); continue
            seg_idx, seg_val = indices[s:e], data[s:e]
            if e - s <= k:
                rows.append(np.full(e - s, i, dtype=np.int32)); cols.append(seg_idx); vals.append(seg_val)
            else:
                top = np.argpartition(-seg_val, k - 1)[:k]
                top = top[np.argsort(-seg_val[top])]
                rows.append(np.full(k, i, dtype=np.int32)); cols.append(seg_idx[top]); vals.append(seg_val[top])
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals).astype(np.float32)
        B = sp.csr_matrix((vals, (rows, cols)), shape=A.shape, dtype=np.float32)
        B = B.maximum(B.T).tocsr(); B.eliminate_zeros(); B.sort_indices()
        return B

    _module_label = "Module 7 · Graph Construction"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 7 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")

    # Optional overall progress bar for M7 (fixed 7 steps)
    _m7_prog = None; _m7_task = None
    if show_internal_progress:
        try:
            from rich.console import Console as _Console
            from rich.progress import Progress as _Progress, SpinnerColumn as _Spin, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time
            _m7_console = _Console(stderr=True)
            if _m7_console.is_terminal:
                _m7_prog = _Progress(_Spin(), _Text(" Build graph"), _Bar(), _Time(), transient=True, console=_m7_console)
                _m7_task = _m7_prog.add_task("graph", total=7)
                _m7_prog.start()
        except Exception:
            _m7_prog = None

    # 1) expression latent
    t0 = _step_start("Compute expression latent (SCVI/SVD)")
    Z_expr, _expr_method, _expr_params = compute_expr_latent_scvi_or_svd(
        adata2,
        n_latent=scvi_n_latent,
        max_epochs=scvi_epochs,
        learning_rate=scvi_lr,
        batch_size=scvi_batch_size,
    )
    adata2.obsm['X_expr'] = Z_expr
    # Record method + params for auditing and print a concise line when progress is on
    try:
        adata2.uns = adata2.uns if isinstance(adata2.uns, dict) else {}
        adata2.uns['module7_expr_latent'] = {"method": _expr_method, "params": _expr_params}
    except Exception:
        pass
    if show_internal_progress:
        try:
            print(f"Module 7 · Expr latent method: {_expr_method}; params={_expr_params}")
        except Exception:
            pass
    _step_done("Compute expression latent (SCVI/SVD)", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # 2) geometry stats
    t0 = _step_start("Impute + scale geometry features")
    coords = adata2.obs[GEOM_FEATS].to_numpy(dtype=np.float32, copy=True)
    coords = SimpleImputer(strategy='median').fit_transform(coords).astype(np.float32)
    Gm = StandardScaler().fit_transform(coords).astype(np.float32)
    adata2.obsm['X_geom'] = Gm
    _step_done("Impute + scale geometry features", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # 3) build graphs
    t0_graph = _step_start("Build expression kNN graph")
    n_obs2 = adata2.n_obs
    k_expr = int(np.clip(k_expr_base if n_obs2 < 80_000 else 16, 10, 30))
    k_geom = int(np.clip(k_geom_base if n_obs2 < 80_000 else 16, 10, 30))
    A_ex = _build_knn_connectivities_by_scanpy(adata2, 'X_expr', n_neighbors=k_expr)
    _step_done("Build expression kNN graph", t0_graph)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass
    t0_graph2 = _step_start("Build geometry kNN graph")
    A_gm = _build_knn_connectivities_by_scanpy(adata2, 'X_geom', n_neighbors=k_geom)
    _step_done("Build geometry kNN graph", t0_graph2)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # 4) adaptive alpha fusion + same-cell downweight
    t0 = _step_start("Fuse graphs + same-cell downweight")
    inter = (A_ex.multiply(A_gm)).astype(bool).sum(1).A.ravel()
    deg_ex = A_ex.getnnz(axis=1); deg_gm = A_gm.getnnz(axis=1)
    union = np.maximum(deg_ex + deg_gm - inter, 1)
    alpha_vec = np.clip(inter / union, 0.15, 0.85).astype(np.float32)
    D_alpha = diags(alpha_vec, 0, dtype=np.float32)
    D_alpha1 = diags(1.0 - alpha_vec, 0, dtype=np.float32)
    A_mix = (W_EXPR_GRAPH * D_alpha @ A_ex + W_GEOM_GRAPH * D_alpha1 @ A_gm).tocsr()
    A_mix = ((A_mix + A_mix.T) * 0.5).tocsr(); A_mix.eliminate_zeros(); A_mix.sort_indices()
    if 'rep_cell_id' in adata2.obs.columns and BETA_SAMECELL != 1.0:
        cid = adata2.obs['rep_cell_id'].astype(str).values
        _, cidnum = np.unique(cid, return_inverse=True)
        rows, cols = A_mix.nonzero()
        same = (cidnum[rows] == cidnum[cols])
        A_mix.data[same] *= float(BETA_SAMECELL)
    _step_done("Fuse graphs + same-cell downweight", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    t0 = _step_start("Row top-k pruning (k=TOPK)")
    A_pruned = _row_topk_csr(A_mix, TOPK)
    _step_done("Row top-k pruning (k=TOPK)", t0)
    if _m7_prog and _m7_task is not None:
        try: _m7_prog.advance(_m7_task, 1)
        except Exception: pass

    # Save degree histogram visualization (intermediate)
    try:
        t_vis = _time.perf_counter()
        import matplotlib.pyplot as plt
        deg = np.asarray(A_pruned.getnnz(axis=1)).ravel()
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        ax.hist(deg, bins=50, color="#4e79a7")
        ax.set_xlabel("Graph degree per node")
        ax.set_ylabel("count")
        plt.tight_layout()
        _save_fig(fig, out_dir or os.getcwd(), "module7_degree_hist")
        plt.close(fig)
        if show_internal_progress:
            print(f"<< Done:  Module 7 · Visualize degree distribution in {_time.perf_counter()-t_vis:0.2f}s")
    except Exception:
        pass

    # normalization
    t0 = _step_start("Normalize adjacency + scale features")
    if EDGE_NORM == 'rw':
        row_sum = np.array(A_pruned.sum(1)).ravel().astype(np.float32); row_sum[row_sum <= 0] = 1.0
        D_inv = diags((1.0 / row_sum).astype(np.float32))
        A_norm = (D_inv @ A_pruned).astype(np.float32)
    elif EDGE_NORM == 'gcn':
        from scipy.sparse import eye
        A_hat = (A_pruned + eye(A_pruned.shape[0], dtype=np.float32)).tocsr()
        d = np.array(A_hat.sum(1)).ravel().astype(np.float32) + 1e-8
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D_inv_sqrt = diags(d_inv_sqrt.astype(np.float32), 0)
        A_norm = (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)
    else:
        A_norm = A_pruned.astype(np.float32)
    A_norm.eliminate_zeros(); A_norm.sort_indices()
    adata2.obsp['A_norm'] = A_norm
    if feature_scale_mode == 'standard':
        adata2.obsm['X_expr_scaled'] = StandardScaler().fit_transform(adata2.obsm['X_expr']).astype(np.float32)
        adata2.obsm['X_geom_scaled'] = StandardScaler().fit_transform(adata2.obsm['X_geom']).astype(np.float32)
    else:
        adata2.obsm['X_expr_scaled'] = adata2.obsm['X_expr'].astype(np.float32, copy=False)
        adata2.obsm['X_geom_scaled'] = adata2.obsm['X_geom'].astype(np.float32, copy=False)
    _step_done("Normalize adjacency + scale features", t0)
    if _m7_prog and _m7_task is not None:
        try:
            _m7_prog.advance(_m7_task, 1)
            _m7_prog.stop()
        except Exception:
            pass
    return adata2


def _build_groupwise_corruption_plan(
    group_values: Sequence[Any],
) -> Dict[str, np.ndarray]:
    """Precompute indices for a random non-self permutation within groups.

    Missing group labels are treated as separate singleton groups so that they
    can never leak features between cells.
    """
    values = np.asarray(group_values, dtype=object).reshape(-1).copy()
    missing = pd.isna(values)
    for index in np.flatnonzero(missing):
        values[index] = f"__cellscope_missing_group_{int(index)}"
    codes, _ = pd.factorize(values, sort=False)
    codes = np.asarray(codes, dtype=np.int64)
    order = np.argsort(codes, kind="stable").astype(np.int64, copy=False)
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size, dtype=np.int64)
    n_groups = int(codes.max()) + 1 if codes.size else 0
    sizes = np.bincount(codes, minlength=n_groups).astype(np.int64)
    starts = np.zeros(n_groups, dtype=np.int64)
    if n_groups > 1:
        starts[1:] = np.cumsum(sizes[:-1], dtype=np.int64)
    return {
        "codes": codes,
        "order": order,
        "position": position,
        "starts": starts,
        "sizes": sizes,
        "missing_count": np.asarray([int(missing.sum())], dtype=np.int64),
    }


def _sample_local_contrastive_pairs(
    adjacency: sp.spmatrix,
    group_values: Optional[Sequence[Any]],
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positive graph edges and non-edge negatives for local DGI loss.

    When group labels are supplied, negatives are restricted to the source
    node's group. Pairs without an eligible within-group non-neighbor are
    skipped instead of silently crossing cell boundaries.
    """
    requested = max(0, int(n_samples))
    graph = adjacency.tocsr().astype(np.float32)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    upper = sp.triu(graph, k=1, format="coo")
    if group_values is None:
        codes = np.zeros(graph.shape[0], dtype=np.int64)
        members = [np.arange(graph.shape[0], dtype=np.int64)]
    else:
        plan = _build_groupwise_corruption_plan(group_values)
        codes = np.asarray(plan["codes"], dtype=np.int64)
        if codes.size != graph.shape[0]:
            raise ValueError(
                "Local contrastive group labels are not aligned to graph nodes"
            )
        same_group_edge = codes[upper.row] == codes[upper.col]
        upper = sp.coo_matrix(
            (
                upper.data[same_group_edge],
                (upper.row[same_group_edge], upper.col[same_group_edge]),
            ),
            shape=upper.shape,
        )
        members = [
            plan["order"][start:start + size]
            for start, size in zip(plan["starts"], plan["sizes"])
        ]
    if requested == 0 or upper.nnz == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()

    rng = np.random.default_rng(int(seed))
    take = min(requested, int(upper.nnz))
    selected = (
        np.arange(upper.nnz, dtype=np.int64)
        if take == upper.nnz
        else rng.choice(upper.nnz, size=take, replace=False)
    )
    source = np.asarray(upper.row[selected], dtype=np.int64)
    positive = np.asarray(upper.col[selected], dtype=np.int64)

    kept_source: List[int] = []
    kept_positive: List[int] = []
    kept_negative: List[int] = []
    for src, pos in zip(source, positive):
        candidates = members[int(codes[src])]
        if candidates.size <= 1:
            continue
        row_start, row_end = graph.indptr[src], graph.indptr[src + 1]
        neighbors = graph.indices[row_start:row_end]
        negative = -1
        # Sparse graphs normally succeed immediately; bounded random trials
        # avoid materializing a large candidate mask for every sampled edge.
        for _ in range(min(32, max(4, int(candidates.size)))):
            candidate = int(candidates[int(rng.integers(candidates.size))])
            if candidate == src:
                continue
            location = np.searchsorted(neighbors, candidate)
            if location >= neighbors.size or neighbors[location] != candidate:
                negative = candidate
                break
        if negative < 0:
            # Deterministic fallback makes small/dense test graphs exact.
            for candidate_value in candidates:
                candidate = int(candidate_value)
                if candidate == src:
                    continue
                location = np.searchsorted(neighbors, candidate)
                if location >= neighbors.size or neighbors[location] != candidate:
                    negative = candidate
                    break
        if negative < 0:
            continue
        kept_source.append(int(src))
        kept_positive.append(int(pos))
        kept_negative.append(negative)

    return (
        np.asarray(kept_source, dtype=np.int64),
        np.asarray(kept_positive, dtype=np.int64),
        np.asarray(kept_negative, dtype=np.int64),
    )


def _run_module8_cell_aware_dmgi(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    *,
    module8_config: Mapping[str, Any],
    feature_keys: Sequence[str],
    output_key: str,
    compatibility_alias: bool,
    broadcast_to_adata1: bool,
    out_dir: Optional[str],
    progress_callback: Optional[Callable[[str], None]],
    show_internal_progress: bool,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Train the MetaSpot-only CA-DMGI backbone."""
    if broadcast_to_adata1:
        raise ValueError(
            "module8.dgi_framework='cell_aware_dmgi' requires native "
            "MetaSpot execution (module8.dgi_unit='metaspot')"
        )
    expression_graph_key = str(
        module8_config.get(
            "dmgi_expression_graph_key",
            "metaspot_expr_connectivities",
        )
    ).strip()
    spatial_graph_key = str(
        module8_config.get(
            "dmgi_spatial_graph_key",
            "metaspot_spatial_connectivities",
        )
    ).strip()
    missing_graphs = [
        key
        for key in (expression_graph_key, spatial_graph_key)
        if key not in adata2.obsp
    ]
    if missing_graphs:
        raise KeyError(f"Missing CA-DMGI relation graphs: {missing_graphs}")
    if "cell_id" not in adata2.obs.columns:
        raise KeyError("CA-DMGI requires adata.obs['cell_id']")
    missing_features = [
        key for key in feature_keys if key not in adata2.obsm
    ]
    if missing_features:
        raise KeyError(f"Missing CA-DMGI node features: {missing_features}")
    feature_arrays = [
        np.asarray(adata2.obsm[key], dtype=np.float32)
        for key in feature_keys
    ]
    if any(array.shape[0] != adata2.n_obs for array in feature_arrays):
        raise ValueError("CA-DMGI node features are not aligned to observations")
    features = np.concatenate(feature_arrays, axis=1).astype(
        np.float32,
        copy=False,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if progress_callback:
        progress_callback(
            "Module 8 · CA-DMGI relation-specific expression/spatial training"
        )

    requested_training_mode = str(
        module8_config.get("training_mode", "auto")
    ).strip().lower().replace("-", "_")
    execution_plan = adata2.uns.get("execution_plan", {})
    if requested_training_mode == "auto":
        effective_training_mode = str(
            execution_plan.get("dgi_training_mode", "full_batch")
        ).strip().lower().replace("-", "_")
    else:
        effective_training_mode = requested_training_mode
    if effective_training_mode not in {"full_batch", "neighbor_sampled"}:
        raise ValueError(
            "module8.training_mode must resolve to 'full_batch' or "
            "'neighbor_sampled'"
        )
    effective_module8_config = dict(module8_config)
    effective_module8_config["training_mode"] = effective_training_mode
    if effective_training_mode == "neighbor_sampled":
        effective_module8_config["dgi_epochs"] = int(
            module8_config.get(
                "sampled_dgi_epochs",
                module8_config.get("dgi_epochs", 100),
            )
        )
        if out_dir:
            effective_module8_config["sampled_checkpoint_dir"] = os.path.join(
                _intermediate_dir(out_dir), "module8_sampled_checkpoints"
            )
        from .sampled_multiplex_dgi import (
            train_sampled_cell_aware_dmgi as train_cell_aware_dmgi,
        )
    else:
        from .multiplex_dgi import train_cell_aware_dmgi

    def _training_progress(epoch: int, epochs: int, loss: float) -> None:
        _logging.getLogger("cellscope").info(
            "[CA-DMGI] epoch %d/%d loss=%.4f",
            epoch,
            epochs,
            loss,
        )

    result = train_cell_aware_dmgi(
        features,
        adata2.obsp[expression_graph_key],
        adata2.obsp[spatial_graph_key],
        adata2.obs["cell_id"].astype(str).to_numpy(),
        config=effective_module8_config,
        progress_callback=_training_progress,
    )
    consensus = np.asarray(result["embedding"], dtype=np.float32)
    expression = np.asarray(
        result["expression_embedding"], dtype=np.float32
    )
    spatial = np.asarray(result["spatial_embedding"], dtype=np.float32)
    expression_gate = np.asarray(
        result["expression_gate"], dtype=np.float32
    )
    adata2.obsm[output_key] = consensus
    adata2.obsm[METASPOT_DMGI_EXPRESSION_KEY] = expression
    adata2.obsm[METASPOT_DMGI_SPATIAL_KEY] = spatial
    adata2.obs = adata2.obs.copy()
    adata2.obs["dmgi_expression_gate"] = expression_gate
    adata2.obs["dmgi_spatial_gate"] = 1.0 - expression_gate
    if compatibility_alias and output_key != "X_dgi_gcn":
        adata2.obsm["X_dgi_gcn"] = consensus.copy()

    history = result["history"]
    metrics = result["metrics"]
    runtime = result["runtime"]
    provenance = dict(adata2.uns.get("dgi_embeddings", {}))
    provenance[output_key] = {
        "schema_version": int(METASPOT_DGI_SCHEMA_VERSION),
        "framework": "cell_aware_dmgi",
        "training_mode_requested": requested_training_mode,
        "training_mode": effective_training_mode,
        "objective": "relation_specific_local_infomax",
        "node_unit": "metaspot",
        "node_id_checksum": _obs_names_checksum(adata2.obs_names),
        "graph_key": "multiplex",
        "expression_graph_key": expression_graph_key,
        "spatial_graph_key": spatial_graph_key,
        "feature_keys": [str(key) for key in feature_keys],
        "n_nodes": int(adata2.n_obs),
        "n_features": int(features.shape[1]),
        "embedding_dim": int(consensus.shape[1]),
        "expression_embedding_key": METASPOT_DMGI_EXPRESSION_KEY,
        "spatial_embedding_key": METASPOT_DMGI_SPATIAL_KEY,
        "gate_obs_key": "dmgi_expression_gate",
        "encoder": str(module8_config.get("dgi_encoder", "residual")),
        "message_passing_layers": int(
            module8_config.get("dgi_layers", 2)
        ),
        "hidden_dim": int(module8_config.get("dgi_hidden", 64)),
        "dropout": float(module8_config.get("dgi_dropout", 0.1)),
        "residual_alpha": float(
            module8_config.get("dgi_residual_alpha", 0.1)
        ),
        "layer_norm": bool(module8_config.get("dgi_layer_norm", True)),
        "corruption_mode_effective": "same_cell",
        "negative_semantics": "corrupted_features_not_real_nodes",
        "expression_loss_weight": float(
            module8_config.get("dmgi_expression_weight", 1.0)
        ),
        "spatial_loss_weight": float(
            module8_config.get("dmgi_spatial_weight", 1.0)
        ),
        "consensus_loss_weight": float(
            module8_config.get("dmgi_consensus_weight", 0.1)
        ),
        "gate_floor": float(module8_config.get("dmgi_gate_floor", 0.1)),
        "epochs_requested": int(
            effective_module8_config.get("dgi_epochs", 100)
        ),
        "epochs": int(metrics["epochs_run"]),
        "best_epoch": int(metrics["best_epoch"]),
        "early_stopped": bool(metrics["early_stopped"]),
        "learning_rate": float(module8_config.get("dgi_lr", 1e-3)),
        "weight_decay": float(
            module8_config.get("dgi_weight_decay", 1e-4)
        ),
        "gradient_clip": float(
            module8_config.get("dgi_clip_grad", 1.0)
        ),
        "random_seed": int(module8_config.get("random_seed", 42)),
        "training_loss": history["loss"],
        "validation_loss": history.get("validation_loss", []),
        "training_expression_infomax_loss": history[
            "expression_infomax_loss"
        ],
        "training_spatial_infomax_loss": history[
            "spatial_infomax_loss"
        ],
        "training_consensus_loss": history["consensus_loss"],
        "training_loss_ema": history["loss_ema"],
        "training_expression_gate_mean": history[
            "expression_gate_mean"
        ],
        "final_training_loss": float(history["loss"][-1]),
        "final_expression_infomax_loss": float(
            history["expression_infomax_loss"][-1]
        ),
        "final_spatial_infomax_loss": float(
            history["spatial_infomax_loss"][-1]
        ),
        "final_consensus_loss": float(history["consensus_loss"][-1]),
        **{str(key): value for key, value in metrics.items()},
        **{str(key): value for key, value in runtime.items()},
        "compatibility_alias": (
            "X_dgi_gcn"
            if compatibility_alias and output_key != "X_dgi_gcn"
            else ""
        ),
    }
    adata2.uns["dgi_embeddings"] = provenance

    training_history = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["loss"]) + 1),
            "total_loss": history["loss"],
            "validation_loss": history.get(
                "validation_loss",
                np.full(len(history["loss"]), np.nan, dtype=np.float32),
            ),
            "expression_infomax_loss": history[
                "expression_infomax_loss"
            ],
            "spatial_infomax_loss": history["spatial_infomax_loss"],
            "consensus_loss": history["consensus_loss"],
            "total_loss_ema": history["loss_ema"],
            "mean_expression_gate": history["expression_gate_mean"],
        }
    )
    summary_fields = {
        "framework": "cell_aware_dmgi",
        "training_mode_requested": requested_training_mode,
        "training_mode": effective_training_mode,
        "objective": "relation_specific_local_infomax",
        "negative_semantics": "corrupted_features_not_real_nodes",
        "n_nodes": int(adata2.n_obs),
        "n_features": int(features.shape[1]),
        "embedding_dim": int(consensus.shape[1]),
        "epochs_requested": int(
            effective_module8_config.get("dgi_epochs", 100)
        ),
        "epochs_run": int(metrics["epochs_run"]),
        "best_epoch": int(metrics["best_epoch"]),
        "early_stopped": bool(metrics["early_stopped"]),
        "device": str(runtime["device"]),
        "device_requested": str(runtime["device_requested"]),
        "cuda_device_name": str(runtime["cuda_device_name"]),
        "cpu_threads": int(runtime["cpu_threads"]),
        "sparse_layout": str(runtime["sparse_layout"]),
        "sparse_index_dtype": str(runtime["sparse_index_dtype"]),
        "fused_corruption_pass": bool(
            runtime["fused_corruption_pass"]
        ),
        "tf32_enabled": bool(runtime["tf32_enabled"]),
        "expression_edges_nnz": int(runtime["expression_edges_nnz"]),
        "spatial_edges_nnz": int(runtime["spatial_edges_nnz"]),
        "corruption_group_count": int(runtime["corruption_group_count"]),
        "final_expression_gate_mean": float(
            metrics["final_expression_gate_mean"]
        ),
        "final_expression_gate_std": float(
            metrics["final_expression_gate_std"]
        ),
        "final_relation_cosine": float(metrics["final_relation_cosine"]),
        "final_expression_context_cosine": float(
            metrics["final_expression_context_cosine"]
        ),
        "final_spatial_context_cosine": float(
            metrics["final_spatial_context_cosine"]
        ),
    }
    try:
        _save_dataframe(
            training_history,
            out_dir or os.getcwd(),
            "module8_ca_dmgi_training_history",
            diagnostic=True,
        )
        _save_dataframe(
            pd.DataFrame([summary_fields]),
            out_dir or os.getcwd(),
            "module8_ca_dmgi_summary",
            diagnostic=True,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "CA-DMGI training audit tables skipped: %s",
            exc,
        )

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        epochs_axis = np.arange(1, len(history["loss"]) + 1)
        axes[0].plot(
            epochs_axis,
            history["expression_infomax_loss"],
            label="Expression-relation Infomax",
        )
        axes[0].plot(
            epochs_axis,
            history["spatial_infomax_loss"],
            label="Within-cell spatial Infomax",
        )
        axes[0].plot(
            epochs_axis,
            history["loss_ema"],
            linestyle="--",
            label="Total loss EMA",
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Relation-specific training objectives")
        axes[0].legend(frameon=False)
        axes[1].plot(
            epochs_axis,
            history["expression_gate_mean"],
            color="#4C78A8",
        )
        axes[1].axhline(0.5, color="#999999", linestyle="--")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean expression-relation gate")
        axes[1].set_title("Learned multiplex consensus gate")
        fig.suptitle(
            "Cell-Aware Deep Multiplex Graph Infomax (CA-DMGI) training",
            fontweight="bold",
        )
        fig.tight_layout()
        _save_fig(
            fig,
            out_dir or os.getcwd(),
            "module8_ca_dmgi_training",
        )
        plt.close(fig)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "CA-DMGI training plot skipped: %s", exc
        )
    return adata1, adata2


def run_module8_dgi(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    show_internal_progress: bool = False,
    dgi_epochs: int = 100,
    enable_dgi_sage: bool = False,
    out_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    *,
    graph_key: str = "A_norm",
    feature_keys: Optional[Sequence[str]] = None,
    output_key: str = "X_dgi_gcn",
    sage_output_key: str = "X_dgi_sage",
    broadcast_to_adata1: bool = True,
    compatibility_alias: bool = False,
) -> Tuple[ad.AnnData, ad.AnnData]:
    import numpy as np
    import torch
    if feature_keys is None:
        feature_keys = ("X_expr_scaled", "X_geom_scaled")

    # Load DGI hyperparameters from config. The residual encoder separates
    # representation depth from graph-propagation depth: two message-passing
    # layers remain the default, while input/output projections provide the
    # additional nonlinear capacity without forcing a wider graph receptive
    # field.
    dgi_encoder = "residual"
    dgi_layers = 2
    dgi_hidden = 64
    dgi_out = 32
    dgi_dropout = 0.1
    dgi_residual_alpha = 0.1
    dgi_layer_norm = True
    dgi_framework = "standard_dgi"
    dgi_corruption_mode = "auto"
    dgi_local_loss_weight = 0.2
    dgi_local_temperature = 0.2
    dgi_local_samples = 65536
    dgi_local_graph_key = "auto"
    dgi_cpu_backend = "auto"
    dgi_lr = 1e-3
    dgi_wd = 1e-4
    dgi_clip = 1.0
    dgi_device_requested = "auto"
    dgi_cpu_threads = 0
    dgi_tf32 = True
    dgi_amp_requested = True
    dgi_seed = 42
    _m8: Mapping[str, Any] = {}
    try:
        _cfg = load_params_yaml() or {}
        _m8 = _cfg.get('module8', {}) if _cfg else {}
        if not isinstance(_m8, Mapping):
            raise TypeError("module8 must be a mapping")
        dgi_epochs = int(_m8.get('dgi_epochs', dgi_epochs))
        enable_dgi_sage = bool(_m8.get('enable_dgi_sage', enable_dgi_sage))
        dgi_encoder = str(_m8.get("dgi_encoder", dgi_encoder)).strip().lower()
        dgi_layers = int(_m8.get("dgi_layers", dgi_layers))
        dgi_hidden = int(_m8.get('dgi_hidden', dgi_hidden))
        dgi_out = int(_m8.get('dgi_out', dgi_out))
        dgi_dropout = float(_m8.get("dgi_dropout", dgi_dropout))
        dgi_residual_alpha = float(
            _m8.get("dgi_residual_alpha", dgi_residual_alpha)
        )
        dgi_layer_norm = bool(_m8.get("dgi_layer_norm", dgi_layer_norm))
        dgi_framework = str(
            _m8.get("dgi_framework", dgi_framework)
        ).strip().lower()
        dgi_corruption_mode = str(
            _m8.get("dgi_corruption_mode", dgi_corruption_mode)
        ).strip().lower()
        dgi_local_loss_weight = float(
            _m8.get("dgi_local_loss_weight", dgi_local_loss_weight)
        )
        dgi_local_temperature = float(
            _m8.get("dgi_local_temperature", dgi_local_temperature)
        )
        dgi_local_samples = int(
            _m8.get("dgi_local_samples", dgi_local_samples)
        )
        dgi_local_graph_key = str(
            _m8.get("dgi_local_graph_key", dgi_local_graph_key)
        ).strip()
        dgi_cpu_backend = str(
            _m8.get("dgi_cpu_backend", dgi_cpu_backend)
        ).strip().lower()
        dgi_lr = float(_m8.get('dgi_lr', dgi_lr))
        dgi_wd = float(_m8.get('dgi_weight_decay', dgi_wd))
        dgi_clip = float(_m8.get('dgi_clip_grad', dgi_clip))
        dgi_device_requested = str(
            _m8.get("dgi_device", dgi_device_requested)
        ).strip().lower()
        dgi_cpu_threads = int(
            _m8.get("dgi_cpu_threads", dgi_cpu_threads)
        )
        dgi_tf32 = bool(_m8.get("dgi_tf32", dgi_tf32))
        dgi_amp_requested = bool(_m8.get('amp', dgi_amp_requested))
        dgi_seed = int(_m8.get("random_seed", dgi_seed))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid module8 DGI configuration: {exc}") from exc

    encoder_aliases = {
        "plain": "plain",
        "gcn": "plain",
        "residual": "residual",
        "residual_gcn": "residual",
        "residual-gcn": "residual",
    }
    if dgi_encoder not in encoder_aliases:
        raise ValueError(
            "module8.dgi_encoder must be one of "
            "{'plain', 'residual'}, "
            f"got {dgi_encoder!r}"
        )
    dgi_encoder = encoder_aliases[dgi_encoder]
    framework_aliases = {
        "standard_dgi": "standard_dgi",
        "standard-dgi": "standard_dgi",
        "dgi": "standard_dgi",
        "cell_aware_dmgi": "cell_aware_dmgi",
        "cell-aware-dmgi": "cell_aware_dmgi",
        "ca_dmgi": "cell_aware_dmgi",
        "ca-dmgi": "cell_aware_dmgi",
        "ch_dmgi": "cell_aware_dmgi",  # legacy compatibility only
        "ch-dmgi": "cell_aware_dmgi",  # legacy compatibility only
        "dmgi": "cell_aware_dmgi",
    }
    if dgi_framework not in framework_aliases:
        raise ValueError(
            "module8.dgi_framework must be one of "
            "{'standard_dgi', 'cell_aware_dmgi'}"
        )
    dgi_framework = framework_aliases[dgi_framework]
    corruption_aliases = {
        "auto": "auto",
        "global": "global",
        "global_shuffle": "global",
        "global-shuffle": "global",
        "same_cell": "same_cell",
        "same-cell": "same_cell",
        "within_cell": "same_cell",
        "within-cell": "same_cell",
    }
    if dgi_corruption_mode not in corruption_aliases:
        raise ValueError(
            "module8.dgi_corruption_mode must be one of "
            "{'auto', 'global', 'same_cell'}, "
            f"got {dgi_corruption_mode!r}"
        )
    dgi_corruption_mode = corruption_aliases[dgi_corruption_mode]
    if dgi_epochs < 1:
        raise ValueError("module8.dgi_epochs must be >= 1")
    if dgi_layers < 1:
        raise ValueError("module8.dgi_layers must be >= 1")
    if dgi_hidden < 1 or dgi_out < 1:
        raise ValueError("module8.dgi_hidden and module8.dgi_out must be >= 1")
    if not 0.0 <= dgi_dropout < 1.0:
        raise ValueError("module8.dgi_dropout must be in [0, 1)")
    if not 0.0 <= dgi_residual_alpha <= 1.0:
        raise ValueError("module8.dgi_residual_alpha must be in [0, 1]")
    if dgi_lr <= 0.0 or dgi_wd < 0.0:
        raise ValueError(
            "module8.dgi_lr must be > 0 and dgi_weight_decay must be >= 0"
        )
    if dgi_clip is not None and dgi_clip <= 0.0:
        raise ValueError("module8.dgi_clip_grad must be > 0")
    if dgi_cpu_threads < 0:
        raise ValueError("module8.dgi_cpu_threads must be >= 0")
    if dgi_local_loss_weight < 0.0:
        raise ValueError("module8.dgi_local_loss_weight must be >= 0")
    if dgi_local_temperature <= 0.0:
        raise ValueError("module8.dgi_local_temperature must be > 0")
    if dgi_local_samples < 0:
        raise ValueError("module8.dgi_local_samples must be >= 0")
    if not dgi_local_graph_key:
        raise ValueError("module8.dgi_local_graph_key must not be empty")
    cpu_backend_aliases = {
        "auto": "auto",
        "pyg": "pyg",
        "coo": "pyg",
        "torch_csr": "torch_csr",
        "torch-csr": "torch_csr",
        "csr": "torch_csr",
    }
    if dgi_cpu_backend not in cpu_backend_aliases:
        raise ValueError(
            "module8.dgi_cpu_backend must be one of "
            "{'auto', 'pyg', 'torch_csr'}"
        )
    dgi_cpu_backend = cpu_backend_aliases[dgi_cpu_backend]
    if dgi_framework == "cell_aware_dmgi":
        effective_dmgi_config = {
            **dict(_m8),
            "dgi_framework": dgi_framework,
            "dgi_epochs": int(dgi_epochs),
            "dgi_encoder": dgi_encoder,
            "dgi_layers": int(dgi_layers),
            "dgi_hidden": int(dgi_hidden),
            "dgi_out": int(dgi_out),
            "dgi_dropout": float(dgi_dropout),
            "dgi_residual_alpha": float(dgi_residual_alpha),
            "dgi_layer_norm": bool(dgi_layer_norm),
            "dgi_lr": float(dgi_lr),
            "dgi_weight_decay": float(dgi_wd),
            "dgi_clip_grad": float(dgi_clip),
            "random_seed": int(dgi_seed),
        }
        return _run_module8_cell_aware_dmgi(
            adata1,
            adata2,
            module8_config=effective_dmgi_config,
            feature_keys=tuple(str(key) for key in feature_keys),
            output_key=output_key,
            compatibility_alias=compatibility_alias,
            broadcast_to_adata1=broadcast_to_adata1,
            out_dir=out_dir,
            progress_callback=progress_callback,
            show_internal_progress=show_internal_progress,
        )

    # Standard DGI uses the fused graph. Delay this comparatively expensive
    # PyG conversion and feature copy until after the CA-DMGI early return;
    # CA-DMGI consumes its two native sparse relation graphs directly.
    from .multiplex_dgi import resolve_dgi_device
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.nn.models import DeepGraphInfomax
    from torch_geometric.utils import from_scipy_sparse_matrix

    device = resolve_dgi_device(torch, dgi_device_requested)
    if device.type == "cpu":
        available_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else int(os.cpu_count() or 1)
        )
        torch.set_num_threads(
            dgi_cpu_threads
            if dgi_cpu_threads > 0
            else min(32, max(1, available_cpus))
        )
    elif hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = dgi_tf32
    dgi_amp = dgi_amp_requested and device.type == "cuda"
    if graph_key not in adata2.obsp:
        raise KeyError(f"Missing DGI graph adata.obsp[{graph_key!r}]")
    A_norm = adata2.obsp[graph_key].tocsr()
    edge_index, edge_weight = from_scipy_sparse_matrix(A_norm)
    edge_weight = torch.clamp(
        torch.nan_to_num(
            edge_weight, nan=0.0, posinf=0.0, neginf=0.0
        ),
        min=1e-8,
    )
    missing_features = [
        key for key in feature_keys if key not in adata2.obsm
    ]
    if missing_features:
        raise KeyError(f"Missing DGI node features: {missing_features}")
    feature_arrays = [
        np.asarray(adata2.obsm[key], dtype=np.float32)
        for key in feature_keys
    ]
    if any(array.shape[0] != adata2.n_obs for array in feature_arrays):
        raise ValueError("DGI node features are not aligned to observations")
    X_all = np.concatenate(feature_arrays, axis=1).astype(np.float32)
    x = torch.from_numpy(np.nan_to_num(X_all, nan=0.0)).to(
        dtype=torch.float32,
        device=device,
    )
    x = F.normalize(x, p=2, dim=1)
    ei = edge_index.to(device, non_blocking=True)
    ew = edge_weight.to(device, non_blocking=True).to(torch.float32)
    effective_gcn_backend = (
        "torch_csr"
        if device.type == "cpu" and dgi_cpu_backend in {"auto", "torch_csr"}
        else "pyg"
    )
    np.random.seed(dgi_seed)
    torch.manual_seed(dgi_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(dgi_seed)

    cell_group_key = "cell_id"
    has_cell_groups = cell_group_key in adata2.obs.columns
    if dgi_corruption_mode == "auto":
        effective_corruption_mode = (
            "same_cell"
            if not broadcast_to_adata1 and has_cell_groups
            else "global"
        )
    else:
        effective_corruption_mode = dgi_corruption_mode
    if effective_corruption_mode == "same_cell" and not has_cell_groups:
        raise ValueError(
            "module8.dgi_corruption_mode='same_cell' requires "
            "adata.obs['cell_id'] aligned to DGI graph nodes"
        )

    corruption_plan = None
    corruption_tensors = None
    if effective_corruption_mode == "same_cell":
        corruption_plan = _build_groupwise_corruption_plan(
            adata2.obs[cell_group_key].to_numpy()
        )
        corruption_tensors = {
            key: torch.as_tensor(
                corruption_plan[key], dtype=torch.long, device=device
            )
            for key in ("codes", "order", "position", "starts", "sizes")
        }

    effective_local_graph_key = ""
    local_pair_arrays = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    if dgi_local_loss_weight > 0.0 and dgi_local_samples > 0:
        if dgi_local_graph_key.lower() == "auto":
            effective_local_graph_key = (
                "metaspot_spatial_connectivities"
                if "metaspot_spatial_connectivities" in adata2.obsp
                else graph_key
            )
        else:
            effective_local_graph_key = dgi_local_graph_key
        if effective_local_graph_key not in adata2.obsp:
            raise KeyError(
                "Missing local DGI graph "
                f"adata.obsp[{effective_local_graph_key!r}]"
            )
        local_groups = (
            adata2.obs[cell_group_key].to_numpy()
            if has_cell_groups
            else None
        )
        local_pair_arrays = _sample_local_contrastive_pairs(
            adata2.obsp[effective_local_graph_key],
            local_groups,
            n_samples=dgi_local_samples,
            seed=dgi_seed + 17,
        )
    local_pair_tensors = tuple(
        torch.as_tensor(values, dtype=torch.long, device=device)
        for values in local_pair_arrays
    )

    cpu_gcn_adjacency = None
    if effective_gcn_backend == "torch_csr":
        # GCNConv with ``normalize=False`` computes A^T @ (XW) under PyG's
        # source-to-target edge convention.  The graph is already normalized,
        # so a single CSR SpMM is mathematically equivalent and substantially
        # faster than COO scatter on CPU for million-edge MetaSpot graphs.
        cpu_graph = A_norm.transpose().tocsr().astype(
            np.float32,
            copy=True,
        )
        cpu_graph.data = np.clip(
            np.nan_to_num(
                cpu_graph.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            1e-8,
            None,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            cpu_gcn_adjacency = torch.sparse_csr_tensor(
                torch.from_numpy(cpu_graph.indptr.astype(np.int64, copy=False)),
                torch.from_numpy(cpu_graph.indices.astype(np.int64, copy=False)),
                torch.from_numpy(cpu_graph.data),
                size=cpu_graph.shape,
                dtype=torch.float32,
                device=device,
            )
        _logging.getLogger("cellscope").info(
            "Module 8 CPU GCN backend | requested=%s | effective=%s | "
            "nodes=%d | edges=%d",
            dgi_cpu_backend,
            effective_gcn_backend,
            int(A_norm.shape[0]),
            int(A_norm.nnz),
        )

    class _TorchCSRGCNConv(nn.Module):
        """GCN linear transform plus pre-normalized CSR aggregation."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
            self.bias = nn.Parameter(torch.empty(output_dim))
            # ``GCNConv`` constructs and initializes its internal Linear once,
            # then calls ``GCNConv.reset_parameters`` and initializes it a
            # second time. Mirror that RNG consumption so the optimized CSR
            # backend preserves seeded initialization as well as graph math.
            self.reset_parameters()
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, values, edge_index=None, edge_weight=None):
            if cpu_gcn_adjacency is None:
                raise RuntimeError("CPU CSR GCN adjacency is unavailable")
            transformed = F.linear(values, self.weight, bias=None)
            return torch.sparse.mm(
                cpu_gcn_adjacency,
                transformed,
            ) + self.bias

    class Encoder(nn.Module):
        """Configurable plain or initial-residual graph encoder for DGI."""

        def __init__(
            self,
            in_channels,
            hidden=64,
            out_channels=32,
            conv_type="gcn",
            architecture="residual",
            num_layers=2,
            dropout=0.1,
            residual_alpha=0.1,
            layer_norm=True,
        ):
            super().__init__()
            Conv = {"gcn": GCNConv, "sage": SAGEConv}[conv_type]
            self.architecture = str(architecture)
            self.num_layers = int(num_layers)
            self.conv_type = conv_type
            self.residual_alpha = float(residual_alpha)
            self.drop = nn.Dropout(float(dropout))

            def make_conv(input_dim, output_dim):
                if conv_type == "gcn":
                    if effective_gcn_backend == "torch_csr":
                        return _TorchCSRGCNConv(input_dim, output_dim)
                    return Conv(
                        input_dim,
                        output_dim,
                        normalize=False,
                        improved=True,
                        add_self_loops=False,
                    )
                return Conv(input_dim, output_dim)

            if self.architecture == "plain":
                dimensions = (
                    [int(in_channels)]
                    + [int(hidden)] * max(0, self.num_layers - 1)
                    + [int(out_channels)]
                )
                self.convs = nn.ModuleList(
                    make_conv(dimensions[index], dimensions[index + 1])
                    for index in range(self.num_layers)
                )
                self.activations = nn.ModuleList(
                    nn.PReLU() for _ in range(max(0, self.num_layers - 1))
                )
                self.input_projection = None
                self.output_projection = None
                self.norms = nn.ModuleList()
                self.input_activation = None
            else:
                self.input_projection = nn.Linear(in_channels, hidden)
                self.input_activation = nn.PReLU()
                self.convs = nn.ModuleList(
                    make_conv(hidden, hidden) for _ in range(self.num_layers)
                )
                self.activations = nn.ModuleList(
                    nn.PReLU() for _ in range(self.num_layers)
                )
                self.norms = nn.ModuleList(
                    nn.LayerNorm(hidden) if layer_norm else nn.Identity()
                    for _ in range(self.num_layers)
                )
                self.output_projection = nn.Linear(hidden, out_channels)

        def _convolve(self, conv, values, edge_index, edge_weight):
            if self.conv_type == "gcn":
                return conv(values, edge_index, edge_weight=edge_weight)
            return conv(values, edge_index)

        def forward(self, x, edge_index, edge_weight=None):
            if self.architecture == "plain":
                values = x
                for layer_index, conv in enumerate(self.convs):
                    values = self._convolve(
                        conv,
                        values,
                        edge_index,
                        edge_weight,
                    )
                    if layer_index < self.num_layers - 1:
                        values = self.activations[layer_index](values)
                        values = self.drop(values)
                return values

            initial = self.input_activation(self.input_projection(x))
            values = self.drop(initial)
            for layer_index, conv in enumerate(self.convs):
                previous = values
                message = self._convolve(
                    conv,
                    values,
                    edge_index,
                    edge_weight,
                )
                message = self.activations[layer_index](message)
                message = self.drop(message)
                message = (
                    (1.0 - self.residual_alpha) * message
                    + self.residual_alpha * initial
                )
                values = self.norms[layer_index](previous + message)
            return self.output_projection(values)

        @torch.no_grad()
        def get_embeddings(self, x, edge_index, edge_weight=None):
            self.eval()
            return self.forward(x, edge_index, edge_weight)

    def build_dgi(conv_type="gcn", hidden=dgi_hidden, out_channels=dgi_out):
        enc = Encoder(
            in_channels=x.size(-1),
            hidden=hidden,
            out_channels=out_channels,
            conv_type=conv_type,
            architecture=dgi_encoder,
            num_layers=dgi_layers,
            dropout=dgi_dropout,
            residual_alpha=dgi_residual_alpha,
            layer_norm=dgi_layer_norm,
        ).to(device)
        def corruption(x_in, edge_index, edge_weight=None):
            if effective_corruption_mode == "global":
                permutation = torch.randperm(
                    x_in.size(0), device=x_in.device
                )
            else:
                assert corruption_tensors is not None
                codes = corruption_tensors["codes"]
                order = corruption_tensors["order"]
                position = corruption_tensors["position"]
                starts = corruption_tensors["starts"]
                sizes = corruption_tensors["sizes"]
                shifts = torch.zeros_like(sizes)
                nonsingleton = sizes > 1
                shifts[nonsingleton] = (
                    torch.floor(
                        torch.rand(
                            int(nonsingleton.sum().item()),
                            device=x_in.device,
                        )
                        * (sizes[nonsingleton] - 1).to(torch.float32)
                    ).to(torch.long)
                    + 1
                )
                within_position = position - starts[codes]
                corrupted_position = (
                    starts[codes]
                    + (within_position + shifts[codes]) % sizes[codes]
                )
                permutation = order[corrupted_position]
            return x_in[permutation], edge_index, edge_weight
        model = DeepGraphInfomax(hidden_channels=out_channels, encoder=enc, summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)), corruption=corruption).to(device)
        return model, enc

    def _make_scaler(use_amp: bool):
        # Compatibility wrapper for different torch versions
        try:
            from torch import amp as _amp
            return _amp.GradScaler(enabled=use_amp)
        except Exception:
            try:
                return torch.cuda.amp.GradScaler(enabled=use_amp)
            except Exception:
                class _Dummy:
                    def scale(self, loss): return loss
                    def unscale_(self, opt): pass
                    def step(self, opt): opt.step()
                    def update(self): pass
                return _Dummy()

    def train_dgi(model, epochs: int, lr=dgi_lr, wd=dgi_wd, clip_grad=dgi_clip, use_amp=dgi_amp, tag="GCN"):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # Auto-detect and set medium precision on GPU for performance
        if device.type == "cuda":
            try:
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass
        scaler = _make_scaler(use_amp)
        model.train()
        losses_hist, dgi_losses_hist, local_losses_hist, ema_hist = [], [], [], []
        ema = None
        local_source, local_positive, local_negative = local_pair_tensors

        def _objective():
            pos_z, neg_z, summary = model(x, ei, ew)
            dgi_loss = model.loss(pos_z, neg_z, summary)
            if (
                dgi_local_loss_weight > 0.0
                and local_source.numel() > 0
            ):
                normalized = F.normalize(pos_z, p=2, dim=1)
                positive_similarity = (
                    normalized[local_source]
                    * normalized[local_positive]
                ).sum(dim=1) / dgi_local_temperature
                negative_similarity = (
                    normalized[local_source]
                    * normalized[local_negative]
                ).sum(dim=1) / dgi_local_temperature
                local_loss = F.softplus(
                    negative_similarity - positive_similarity
                ).mean()
            else:
                local_loss = pos_z.sum() * 0.0
            total_loss = dgi_loss + dgi_local_loss_weight * local_loss
            return total_loss, dgi_loss, local_loss
        # Progress bar per-epoch when internal progress enabled
        _m8_console = None; _m8_prog = None; _m8_task = None
        if show_internal_progress:
            try:
                from rich.console import Console as _Console
                from rich.progress import Progress as _Progress, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time, TaskProgressColumn as _TaskProg
                _m8_console = _Console(stderr=True)
                if _m8_console.is_terminal:
                    _m8_prog = _Progress(_Text(f" [bold][M8] Train DGI-{tag}"), _Bar(), _TaskProg(), _Time(), transient=True, console=_m8_console)
                    _m8_task = _m8_prog.add_task(f"DGI-{tag}", total=epochs)
                    _m8_prog.start()
            except Exception:
                _m8_prog = None
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            if use_amp:
                try:
                    from torch import amp as _amp
                    ctx = _amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
                except Exception:
                    ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                with ctx:
                    loss, dgi_loss_value, local_loss_value = _objective()
                try:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                except Exception:
                    loss.backward()
                if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                try:
                    scaler.step(opt); scaler.update()
                except Exception:
                    opt.step()
            else:
                loss, dgi_loss_value, local_loss_value = _objective()
                loss.backward();
                if clip_grad is not None: nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                opt.step()
            if show_internal_progress and ep % 25 == 0:
                print(f"[DGI-{tag}] epoch {ep}/{epochs} loss={float(loss):.4f}")
            if _m8_prog and _m8_task is not None:
                try: _m8_prog.advance(_m8_task, 1)
                except Exception: pass
            # record history
            l = float(loss)
            losses_hist.append(l)
            dgi_losses_hist.append(float(dgi_loss_value))
            local_losses_hist.append(float(local_loss_value))
            ema = l if ema is None else (0.9 * ema + 0.1 * l)
            ema_hist.append(ema)
        # save training curves
        try:
            import matplotlib.pyplot as plt
            import numpy as _np
            fig = plt.figure(figsize=(7,4))
            plt.plot(_np.array(losses_hist), label=f"{tag}-total", linewidth=1.2)
            plt.plot(_np.array(dgi_losses_hist), label=f"{tag}-DGI", linewidth=1.0)
            if dgi_local_loss_weight > 0.0 and local_source.numel() > 0:
                plt.plot(
                    dgi_local_loss_weight * _np.array(local_losses_hist),
                    label=f"{tag}-weighted-local",
                    linewidth=1.0,
                )
            plt.plot(_np.array(ema_hist), label=f"{tag}-ema", linewidth=1.5, linestyle="--")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("DGI Training Curves"); plt.legend(frameon=False); plt.tight_layout()
            _save_fig(fig, out_dir or os.getcwd(), f"module8_dgi_{tag}_curves")
            plt.close(fig)
        except Exception:
            pass
        try:
            if _m8_prog:
                _m8_prog.stop()
        except Exception:
            pass
        history = {
            "loss": np.asarray(losses_hist, dtype=np.float32),
            "dgi_loss": np.asarray(dgi_losses_hist, dtype=np.float32),
            "local_loss": np.asarray(local_losses_hist, dtype=np.float32),
            "loss_ema": np.asarray(ema_hist, dtype=np.float32),
        }
        return model, history

    @torch.no_grad()
    def get_embeddings(encoder):
        z = encoder.get_embeddings(x, ei, ew); z = z.detach().cpu().numpy().astype(np.float32, copy=False); return z

    import time as _time
    _module_label = "Module 8 · Graph Embeddings (DGI)"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 8 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # Train GCN
    t0 = _step_start("Build + train DGI (GCN)")
    dgi_gcn, enc_gcn = build_dgi(conv_type="gcn", hidden=dgi_hidden, out_channels=dgi_out)
    dgi_gcn, gcn_history = train_dgi(
        dgi_gcn,
        epochs=dgi_epochs,
        tag="GCN",
    )
    Z_gcn = get_embeddings(enc_gcn)
    _step_done("Build + train DGI (GCN)", t0)
    del dgi_gcn, enc_gcn
    if device.type == "cuda": torch.cuda.empty_cache();
    # Optional SAGE
    Z_sage = None
    if enable_dgi_sage:
        try:
            t0 = _step_start("Build + train DGI (SAGE)")
            dgi_sage, enc_sage = build_dgi(conv_type="sage", hidden=dgi_hidden, out_channels=dgi_out)
            dgi_sage, _sage_history = train_dgi(
                dgi_sage,
                epochs=dgi_epochs,
                tag="SAGE",
            )
            Z_sage = get_embeddings(enc_sage)
            _step_done("Build + train DGI (SAGE)", t0)
            del dgi_sage, enc_sage
            if device.type == "cuda": torch.cuda.empty_cache()
        except Exception:
            Z_sage = None

    # Attach to the graph's native observation unit.
    attach_label = (
        "Attach native-unit embeddings"
        if not broadcast_to_adata1
        else "Attach embeddings to adata2 + broadcast to adata1"
    )
    t0 = _step_start(attach_label)
    adata2.obsm[output_key] = Z_gcn
    if Z_sage is not None:
        adata2.obsm[sage_output_key] = Z_sage
    if compatibility_alias and output_key != "X_dgi_gcn":
        adata2.obsm["X_dgi_gcn"] = Z_gcn.copy()
    if (
        compatibility_alias
        and Z_sage is not None
        and sage_output_key != "X_dgi_sage"
    ):
        adata2.obsm["X_dgi_sage"] = Z_sage.copy()

    local_positive_cosine = float("nan")
    local_negative_cosine = float("nan")
    local_cosine_margin = float("nan")
    if local_pair_arrays[0].size > 0:
        embedding_norm = np.linalg.norm(Z_gcn, axis=1, keepdims=True)
        normalized_embedding = Z_gcn / np.maximum(embedding_norm, 1e-12)
        local_source_np, local_positive_np, local_negative_np = (
            local_pair_arrays
        )
        positive_cosines = np.sum(
            normalized_embedding[local_source_np]
            * normalized_embedding[local_positive_np],
            axis=1,
        )
        negative_cosines = np.sum(
            normalized_embedding[local_source_np]
            * normalized_embedding[local_negative_np],
            axis=1,
        )
        local_positive_cosine = float(np.mean(positive_cosines))
        local_negative_cosine = float(np.mean(negative_cosines))
        local_cosine_margin = float(
            local_positive_cosine - local_negative_cosine
        )

    node_unit = "metaspot" if not broadcast_to_adata1 else "metadomain"
    provenance = dict(adata2.uns.get("dgi_embeddings", {}))
    provenance[output_key] = {
        "schema_version": int(
            METASPOT_DGI_SCHEMA_VERSION
            if node_unit == "metaspot"
            else 1
        ),
        "node_unit": node_unit,
        "framework": "standard_dgi",
        "objective": "global_dgi_plus_pairwise_local_ranking",
        "node_id_checksum": _obs_names_checksum(adata2.obs_names),
        "graph_key": graph_key,
        "feature_keys": [str(key) for key in feature_keys],
        "n_nodes": int(adata2.n_obs),
        "n_features": int(X_all.shape[1]),
        "embedding_dim": int(Z_gcn.shape[1]),
        "encoder": str(dgi_encoder),
        "message_passing_layers": int(dgi_layers),
        "hidden_dim": int(dgi_hidden),
        "dropout": float(dgi_dropout),
        "residual_alpha": float(dgi_residual_alpha),
        "layer_norm": bool(dgi_layer_norm),
        "corruption_mode_requested": str(dgi_corruption_mode),
        "corruption_mode_effective": str(effective_corruption_mode),
        "corruption_group_key": (
            cell_group_key
            if effective_corruption_mode == "same_cell"
            else ""
        ),
        "corruption_group_count": int(
            len(corruption_plan["sizes"])
            if corruption_plan is not None
            else 0
        ),
        "corruption_singleton_group_count": int(
            np.sum(corruption_plan["sizes"] == 1)
            if corruption_plan is not None
            else 0
        ),
        "local_graph_key": str(effective_local_graph_key),
        "gcn_backend_requested": str(dgi_cpu_backend),
        "gcn_backend_effective": str(effective_gcn_backend),
        "local_loss_weight": float(dgi_local_loss_weight),
        "local_temperature": float(dgi_local_temperature),
        "local_samples_requested": int(dgi_local_samples),
        "local_pairs_sampled": int(local_pair_arrays[0].size),
        "final_local_positive_cosine": local_positive_cosine,
        "final_local_negative_cosine": local_negative_cosine,
        "final_local_cosine_margin": local_cosine_margin,
        "epochs": int(dgi_epochs),
        "learning_rate": float(dgi_lr),
        "weight_decay": float(dgi_wd),
        "gradient_clip": float(dgi_clip),
        "amp": bool(dgi_amp),
        "random_seed": int(dgi_seed),
        "training_loss": gcn_history["loss"],
        "training_dgi_loss": gcn_history["dgi_loss"],
        "training_local_loss": gcn_history["local_loss"],
        "training_loss_ema": gcn_history["loss_ema"],
        "final_training_loss": float(gcn_history["loss"][-1]),
        "final_dgi_loss": float(gcn_history["dgi_loss"][-1]),
        "final_local_loss": float(gcn_history["local_loss"][-1]),
        "compatibility_alias": (
            "X_dgi_gcn"
            if compatibility_alias and output_key != "X_dgi_gcn"
            else ""
        ),
    }
    adata2.uns["dgi_embeddings"] = provenance

    if broadcast_to_adata1:
        if "meta-domain" not in adata1.obs.columns:
            raise KeyError(
                "Legacy MetaDomain DGI broadcast requires "
                "adata1.obs['meta-domain']"
            )
        clu_to_row = {
            clu: i for i, clu in enumerate(adata2.obs_names.astype(str))
        }
        row_idx_adata1 = (
            adata1.obs["meta-domain"].astype(str).map(clu_to_row).to_numpy()
        )

        def _broadcast(Z, row_index):
            out = np.zeros((len(row_index), Z.shape[1]), dtype=np.float32)
            mask = (row_index >= 0) & (row_index < Z.shape[0])
            out[mask] = Z[row_index[mask]]
            return out

        adata1.obsm["X_dgi_gcn"] = _broadcast(Z_gcn, row_idx_adata1)
        if Z_sage is not None:
            adata1.obsm["X_dgi_sage"] = _broadcast(Z_sage, row_idx_adata1)
    _step_done(attach_label, t0)
    return adata1, adata2


def _graph_unit_from_config(config: Dict[str, Any]) -> str:
    return dgi_unit_from_config(config)


def _build_module_execution_order(config: Dict[str, Any]) -> List[str]:
    """Return the dependency order for the selected graph observation unit."""
    if _graph_unit_from_config(config) == "metaspot":
        return ["m1", "m2", "m3", "m4", "m7", "m8", "m5", "m6"]
    return ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]


def build_anndata(spatial_df: pd.DataFrame,
                  cell_df: pd.DataFrame,
                  nucleus_df: pd.DataFrame,
                  show_internal_progress: bool = False,
                  progress_callback: Optional[Callable[[str], None]] = None,
                  out_dir: Optional[str] = None,
                  scvi_epochs: int = 100,
                  dgi_epochs: int = 100,
                  enable_dgi_sage: bool = False,
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
    graph_unit = _graph_unit_from_config(cfg)
    module_order = _build_module_execution_order(cfg)
    fingerprint_order = [m for m in module_order if m != "m4"]
    # Determine the earliest changed module in the actual dependency order.
    start_mod = fingerprint_order[0]
    for m in fingerprint_order:
        if prev_mod_fps.get(m) != cur_mod_fps.get(m):
            start_mod = m
            break
    def _mod_le(m: str, target: str) -> bool:
        return module_order.index(m) <= module_order.index(target)
    def _mod_lt(m: str, target: str) -> bool:
        return module_order.index(m) < module_order.index(target)
    try:
        import logging as _logging
        _logging.getLogger("cellscope").info("Fine-grained resume: start module %s", start_mod)
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
    if graph_unit == "metaspot" and _mod_lt("m4", start_mod):
        _late_point_df_cache = _load_late_point_resume_checkpoint(
            out_dir,
            spatial_df,
        )
        if _late_point_df_cache is not None:
            _logging.getLogger("cellscope").info(
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
                _logging.getLogger("cellscope").info("Resume: using Module 1 cache (module1_point_features)")
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
        _logging.getLogger("cellscope").info("Module 1 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m1_recomputed) else ("recomputed" if _m1_recomputed else "cache")
            progress_callback("Module 1 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 2: composition embeddings (+gmm features)
    _m2_use_point_source = False
    try:
        _cfg = load_params_yaml() or {}
        _m2_cfg = _cfg.get('module2', {}) if _cfg else {}
        _m2_use_point_source = bool(_m2_cfg.get('source_from_point_df', False))
    except Exception:
        pass
    # Module 2 only selects the columns it needs inside per-cell blocks. Passing
    # the original table avoids copying hundreds of millions of rows up front.
    if _m2_use_point_source and {'cell_id', 'x_location', 'y_location', 'feature_name'}.issubset(point_df.columns):
        _m2_source_df = point_df
    else:
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
                _logging.getLogger("cellscope").info("Resume: using Module 2 cache (module2_point_df)")
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
            point_df, gmm_cols, cell_emb = build_composition_embeddings(point_df, _m2_source_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
            if isinstance(cell_emb, _LocalCompositionModel):
                composition_model = cell_emb
                try:
                    import joblib
                    joblib.dump(composition_model, os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib"))
                except Exception:
                    pass
            # Attach cell-level composition embedding for downstream MetaSpot segmentation
            try:
                if isinstance(cell_emb, pd.DataFrame):
                    setattr(point_df, '_cell_comp_emb', cell_emb)
            except Exception:
                pass
            _save_dataframe(point_df, out_dir, "module2_point_df")
            _m2_recomputed = True
    else:
        if progress_callback: progress_callback("Module 2 · Composition Embeddings")
        point_df, gmm_cols, cell_emb = build_composition_embeddings(point_df, _m2_source_df, sampled_cell_ids, show_internal_progress=show_internal_progress)
        if isinstance(cell_emb, _LocalCompositionModel):
            composition_model = cell_emb
            try:
                import joblib
                joblib.dump(composition_model, os.path.join(_intermediate_dir(out_dir), "module2_local_composition.joblib"))
            except Exception:
                pass
        try:
            if isinstance(cell_emb, pd.DataFrame):
                setattr(point_df, '_cell_comp_emb', cell_emb)
        except Exception:
            pass
        _save_dataframe(point_df, out_dir, "module2_point_df")
        _m2_recomputed = True
    if _m2_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m2_recomputed) else ("recomputed" if _m2_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 2 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m2_recomputed) else ("recomputed" if _m2_recomputed else "cache")
            progress_callback("Module 2 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # Module 3: per-cell MetaSpot segmentation
    # Read Module 3 clustering options from YAML
    _m3_hdb_min = 6
    _m3_allow_single = True
    _m3_min_samples_frac = 0.3
    _m3_target_sub_size = 12
    _m3_n_feat_hdb = 24
    _m3_kmeans_min_k = 2
    _m3_kmeans_max_k = 999
    _m3_use_k_cap = False
    _m3_merge_min = None
    _m3_hdbscan_max_points = 512
    _m3_kmeans_train_size = 1024
    _m3_kmeans_batch_size = 2048
    _m3_small_backend_cells = 12000
    _m3_sequential_cells = 256
    _m3_enforce_spatial_connectivity = False
    _m3_partition_by_nucleus = False
    _m3_spatial_connectivity_k = 8
    _m3_spatial_radius_quantile = 0.9
    _m3_spatial_radius_factor = 1.5
    _m3_spatial_smoothing_iterations = 1
    _m3_spatial_smoothing_min_fraction = 0.6
    _m3_spatial_min_component_size = None
    _m3_spatial_compactness = 1.0
    _m3_target_sites_per_metaspot = None
    _m3_min_metaspots_per_cell = 1
    _m3_max_metaspots_per_cell = None
    _m3_parallel_chunk_cells = None
    _m3_parallel_n_jobs = None
    try:
        _cfg = load_params_yaml()
        _m3 = _cfg.get('module3', {}) if _cfg else {}
        if isinstance(_m3.get('hdbscan_min_cluster_size'), (int, float)):
            _m3_hdb_min = int(_m3.get('hdbscan_min_cluster_size'))
        if 'hdbscan_allow_single_cluster' in _m3:
            _m3_allow_single = bool(_m3.get('hdbscan_allow_single_cluster'))
        if isinstance(_m3.get('hdbscan_min_samples_frac'), (int, float)):
            _m3_min_samples_frac = float(_m3.get('hdbscan_min_samples_frac'))
        if isinstance(_m3.get('target_sub_size'), (int, float)):
            _m3_target_sub_size = int(_m3.get('target_sub_size')) if _m3.get('target_sub_size') is not None else _m3_target_sub_size
        if isinstance(_m3.get('n_feat_hdb'), (int, float)):
            _m3_n_feat_hdb = int(_m3.get('n_feat_hdb')) if _m3.get('n_feat_hdb') is not None else _m3_n_feat_hdb
        if isinstance(_m3.get('kmeans_min_k'), (int, float)):
            _m3_kmeans_min_k = int(_m3.get('kmeans_min_k'))
        if isinstance(_m3.get('kmeans_max_k'), (int, float)):
            _m3_kmeans_max_k = int(_m3.get('kmeans_max_k'))
        if 'use_k_cap' in _m3:
            _m3_use_k_cap = bool(_m3.get('use_k_cap'))
        # allow null (None) to mean: use MIN_SUB_PTS for merging
        if 'merge_min_size' in _m3:
            val = _m3.get('merge_min_size')
            _m3_merge_min = int(val) if isinstance(val, (int, float)) else None
        if isinstance(_m3.get('hdbscan_max_points'), (int, float)):
            _m3_hdbscan_max_points = int(_m3.get('hdbscan_max_points'))
        if isinstance(_m3.get('kmeans_train_size'), (int, float)):
            _m3_kmeans_train_size = int(_m3.get('kmeans_train_size'))
        if isinstance(_m3.get('kmeans_batch_size'), (int, float)):
            _m3_kmeans_batch_size = int(_m3.get('kmeans_batch_size'))
        if isinstance(_m3.get('small_backend_cells'), (int, float)):
            _m3_small_backend_cells = int(_m3.get('small_backend_cells'))
        if isinstance(_m3.get('sequential_cells'), (int, float)):
            _m3_sequential_cells = int(_m3.get('sequential_cells'))
        if 'enforce_spatial_connectivity' in _m3:
            _m3_enforce_spatial_connectivity = bool(_m3.get('enforce_spatial_connectivity'))
        if 'partition_by_nucleus' in _m3:
            _m3_partition_by_nucleus = bool(
                _m3.get('partition_by_nucleus')
            )
        if isinstance(_m3.get('spatial_connectivity_k'), (int, float)):
            _m3_spatial_connectivity_k = int(_m3.get('spatial_connectivity_k'))
        if isinstance(_m3.get('spatial_radius_quantile'), (int, float)):
            _m3_spatial_radius_quantile = float(_m3.get('spatial_radius_quantile'))
        if isinstance(_m3.get('spatial_radius_factor'), (int, float)):
            _m3_spatial_radius_factor = float(_m3.get('spatial_radius_factor'))
        if isinstance(_m3.get('spatial_smoothing_iterations'), (int, float)):
            _m3_spatial_smoothing_iterations = int(_m3.get('spatial_smoothing_iterations'))
        if isinstance(_m3.get('spatial_smoothing_min_fraction'), (int, float)):
            _m3_spatial_smoothing_min_fraction = float(_m3.get('spatial_smoothing_min_fraction'))
        if 'spatial_min_component_size' in _m3:
            val = _m3.get('spatial_min_component_size')
            _m3_spatial_min_component_size = int(val) if isinstance(val, (int, float)) else None
        if isinstance(_m3.get('spatial_compactness'), (int, float)):
            _m3_spatial_compactness = float(_m3.get('spatial_compactness'))
        if isinstance(_m3.get('target_sites_per_metaspot'), (int, float)):
            _m3_target_sites_per_metaspot = int(_m3.get('target_sites_per_metaspot'))
        if isinstance(_m3.get('min_metaspots_per_cell'), (int, float)):
            _m3_min_metaspots_per_cell = int(_m3.get('min_metaspots_per_cell'))
        if isinstance(_m3.get('max_metaspots_per_cell'), (int, float)):
            _m3_max_metaspots_per_cell = int(_m3.get('max_metaspots_per_cell'))
        if isinstance(_m3.get('parallel_chunk_cells'), (int, float)):
            _m3_parallel_chunk_cells = int(_m3.get('parallel_chunk_cells'))
        if isinstance(_m3.get('parallel_n_jobs'), (int, float)):
            _m3_parallel_n_jobs = int(_m3.get('parallel_n_jobs'))
    except Exception:
        pass
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
        _logging.getLogger("cellscope").info(
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
            hdbscan_min_cluster_size=_m3_hdb_min,
            hdbscan_allow_single_cluster=_m3_allow_single,
            hdbscan_min_samples_frac=_m3_min_samples_frac,
            target_sub_size=_m3_target_sub_size,
            n_feat_hdb=_m3_n_feat_hdb,
            kmeans_min_k=_m3_kmeans_min_k,
            kmeans_max_k=_m3_kmeans_max_k,
            use_k_cap=_m3_use_k_cap,
            merge_min_size=_m3_merge_min,
            hdbscan_max_points=_m3_hdbscan_max_points,
            kmeans_train_size=_m3_kmeans_train_size,
            kmeans_batch_size=_m3_kmeans_batch_size,
            small_backend_cells=_m3_small_backend_cells,
            sequential_cells=_m3_sequential_cells,
            enforce_spatial_connectivity=_m3_enforce_spatial_connectivity,
            partition_by_nucleus=_m3_partition_by_nucleus,
            spatial_connectivity_k=_m3_spatial_connectivity_k,
            spatial_radius_quantile=_m3_spatial_radius_quantile,
            spatial_radius_factor=_m3_spatial_radius_factor,
            spatial_smoothing_iterations=_m3_spatial_smoothing_iterations,
            spatial_smoothing_min_fraction=_m3_spatial_smoothing_min_fraction,
            spatial_min_component_size=_m3_spatial_min_component_size,
            spatial_compactness=_m3_spatial_compactness,
            target_sites_per_metaspot=_m3_target_sites_per_metaspot,
            min_metaspots_per_cell=_m3_min_metaspots_per_cell,
            max_metaspots_per_cell=_m3_max_metaspots_per_cell,
            parallel_chunk_cells=_m3_parallel_chunk_cells,
            parallel_n_jobs=_m3_parallel_n_jobs,
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
        _logging.getLogger("cellscope").info(
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
    _logging.getLogger("cellscope").info(
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
                partition_by_nucleus=_m3_partition_by_nucleus,
            )
            _save_dataframe(
                segmentation_audit,
                out_dir,
                "module3_metaspot_segmentation_audit",
                diagnostic=True,
            )
        _logging.getLogger("cellscope").info(
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
        _logging.getLogger("cellscope").info(
            "Module 3 segmentation audit done in %.2fs",
            _time.perf_counter() - _audit_t0,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
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
    execution_plan: Optional[Dict[str, Any]] = None
    if graph_unit == "metaspot":
        from .scalable_execution import resolve_execution_plan

        resolved_plan = resolve_execution_plan(
            cfg,
            n_metaspots=int(point_df[METASPOT_COL].nunique()),
            n_transcripts=int(len(point_df)),
        )
        execution_plan = resolved_plan.to_dict()
        if not bool(execution_plan.get("full_diagnostics", True)):
            _logging.getLogger("cellscope").info(
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
                _logging.getLogger("cellscope").warning(
                    "Rejecting Module 4 cache: %s",
                    alignment_reason,
                )
                a1_cache = None
        if a1_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 4 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 4 cache (module4_adata1.h5ad)")
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
    _logging.getLogger("cellscope").info(
        "Module 4 MetaSpot alignment validated (%d MetaSpots)",
        adata1.n_obs,
    )
    if _m4_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m4_recomputed) else ("recomputed" if _m4_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 4 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m4_recomputed) else ("recomputed" if _m4_recomputed else "cache")
            progress_callback("Module 4 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    if graph_unit == "metaspot":
        if execution_plan is None:
            raise RuntimeError("MetaSpot execution plan was not resolved")
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
            _logging.getLogger("cellscope").warning(
                "Execution-plan manifest could not be written: %s",
                exc,
            )
        _logging.getLogger("cellscope").info(
            "Scale-aware execution plan | profile=%s | MetaSpots=%d | "
            "kNN=%s | DGI=%s | MetaDomain=%s",
            execution_plan["effective_profile"],
            int(adata1.n_obs),
            execution_plan["knn_backend"],
            execution_plan["dgi_training_mode"],
            execution_plan["metadomain_method"],
        )

    # In MetaSpot graph mode, Modules 7 and 8 are true upstream dependencies
    # of MetaDomain construction.  Keep separate checkpoints so an old
    # MetaDomain-broadcast embedding can never masquerade as a MetaSpot model.
    if graph_unit == "metaspot":
        module7_config = (cfg.get("module7", {}) or {})
        neighbor_config = module7_config.get("neighbors", {}) or {}
        if not isinstance(neighbor_config, dict):
            neighbor_config = {}

        _m7_metaspot_recomputed = False
        if _mod_lt("m7", start_mod) and _cache_chain_valid:
            graph_cache = _load_adata_checkpoint(
                os.path.join(
                    _intermediate_dir(out_dir),
                    "module7_metaspot_adata1.h5ad",
                )
            )
            if graph_cache is not None:
                valid, reason = _metaspot_cache_matches(
                    graph_cache,
                    adata1,
                    required_obsm=("X_expr_metaspot_scaled",),
                    required_obsp=(METASPOT_GRAPH_KEY,),
                )
                graph_meta = graph_cache.uns.get("metaspot_graph", {})
                valid = bool(
                    valid
                    and graph_meta.get("node_unit") == "metaspot"
                    and int(graph_meta.get("schema_version", -1))
                    == METASPOT_GRAPH_SCHEMA_VERSION
                    and graph_meta.get("execution_profile")
                    == execution_plan["effective_profile"]
                    and graph_meta.get("knn_backend")
                    == execution_plan["knn_backend"]
                )
                if not valid:
                    _logging.getLogger("cellscope").warning(
                        "Rejecting MetaSpot graph cache: %s",
                        reason,
                    )
                    graph_cache = None
            if graph_cache is not None:
                if progress_callback:
                    progress_callback("Resume: load Module 7 MetaSpot graph")
                adata1 = graph_cache
            else:
                if progress_callback:
                    progress_callback("Module 7 · MetaSpot Graph Construction")
                adata1 = build_metaspot_graph(
                    adata1,
                    point_df,
                    k_expr=int(neighbor_config.get("k_expr", 15)),
                    k_spatial=int(neighbor_config.get("k_spatial", 6)),
                    expr_n_components=int(
                        module7_config.get(
                            "metaspot_expr_n_components",
                            module7_config.get("scvi_n_latent", 32),
                        )
                    ),
                    alpha_expr=float(
                        module7_config.get("metaspot_alpha_expr", 0.8)
                    ),
                    alpha_spatial=float(
                        module7_config.get("metaspot_alpha_spatial", 0.2)
                    ),
                    edge_norm=str(
                        module7_config.get("metaspot_edge_norm", "gcn")
                    ),
                    use_geometry_features=bool(
                        module7_config.get(
                            "metaspot_use_geometry_features",
                            False,
                        )
                    ),
                    expression_cross_cell_only=bool(
                        module7_config.get(
                            "metaspot_expr_cross_cell_only",
                            False,
                        )
                    ),
                    max_full_batch_nodes=int(
                        module7_config.get(
                            "max_full_batch_nodes",
                            200_000,
                        )
                    ),
                    execution_profile=str(
                        execution_plan["effective_profile"]
                    ),
                    knn_backend=str(execution_plan["knn_backend"]),
                    ann_candidate_multiplier=int(
                        module7_config.get("ann_candidate_multiplier", 4)
                    ),
                    ann_query_chunk_nodes=int(
                        module7_config.get("ann_query_chunk_nodes", 100_000)
                    ),
                    ann_train_sample=int(
                        module7_config.get("ann_train_sample", 1_000_000)
                    ),
                    faiss_hnsw_m=int(
                        module7_config.get("faiss_hnsw_m", 32)
                    ),
                    faiss_nprobe=int(
                        module7_config.get("faiss_nprobe", 32)
                    ),
                    faiss_pq_subvectors=int(
                        module7_config.get("faiss_pq_subvectors", 8)
                    ),
                    faiss_pq_bits=int(
                        module7_config.get("faiss_pq_bits", 8)
                    ),
                    faiss_use_gpu=module7_config.get(
                        "faiss_use_gpu", "auto"
                    ),
                    faiss_gpu_device=int(
                        module7_config.get("faiss_gpu_device", 0)
                    ),
                )
                _save_adata(adata1, out_dir, "module7_metaspot_adata1")
                _m7_metaspot_recomputed = True
        else:
            if progress_callback:
                progress_callback("Module 7 · MetaSpot Graph Construction")
            adata1 = build_metaspot_graph(
                adata1,
                point_df,
                k_expr=int(neighbor_config.get("k_expr", 15)),
                k_spatial=int(neighbor_config.get("k_spatial", 6)),
                expr_n_components=int(
                    module7_config.get(
                        "metaspot_expr_n_components",
                        module7_config.get("scvi_n_latent", 32),
                    )
                ),
                alpha_expr=float(
                    module7_config.get("metaspot_alpha_expr", 0.8)
                ),
                alpha_spatial=float(
                    module7_config.get("metaspot_alpha_spatial", 0.2)
                ),
                edge_norm=str(
                    module7_config.get("metaspot_edge_norm", "gcn")
                ),
                use_geometry_features=bool(
                    module7_config.get(
                        "metaspot_use_geometry_features",
                        False,
                    )
                ),
                expression_cross_cell_only=bool(
                    module7_config.get(
                        "metaspot_expr_cross_cell_only",
                        False,
                    )
                ),
                max_full_batch_nodes=int(
                    module7_config.get("max_full_batch_nodes", 200_000)
                ),
                execution_profile=str(execution_plan["effective_profile"]),
                knn_backend=str(execution_plan["knn_backend"]),
                ann_candidate_multiplier=int(
                    module7_config.get("ann_candidate_multiplier", 4)
                ),
                ann_query_chunk_nodes=int(
                    module7_config.get("ann_query_chunk_nodes", 100_000)
                ),
                ann_train_sample=int(
                    module7_config.get("ann_train_sample", 1_000_000)
                ),
                faiss_hnsw_m=int(module7_config.get("faiss_hnsw_m", 32)),
                faiss_nprobe=int(module7_config.get("faiss_nprobe", 32)),
                faiss_pq_subvectors=int(
                    module7_config.get("faiss_pq_subvectors", 8)
                ),
                faiss_pq_bits=int(module7_config.get("faiss_pq_bits", 8)),
                faiss_use_gpu=module7_config.get("faiss_use_gpu", "auto"),
                faiss_gpu_device=int(
                    module7_config.get("faiss_gpu_device", 0)
                ),
            )
            _save_adata(adata1, out_dir, "module7_metaspot_adata1")
            _m7_metaspot_recomputed = True
        if _m7_metaspot_recomputed:
            _cache_chain_valid = False
        _logging.getLogger("cellscope").info(
            "Module 7 MetaSpot graph ready: nodes=%d edges=%d",
            adata1.n_obs,
            int(adata1.obsp[METASPOT_GRAPH_KEY].nnz),
        )

        _m8_metaspot_recomputed = False
        if _mod_lt("m8", start_mod) and _cache_chain_valid:
            dgi_cache = _load_adata_checkpoint(
                os.path.join(
                    _intermediate_dir(out_dir),
                    "module8_metaspot_adata1.h5ad",
                )
            )
            if dgi_cache is not None:
                valid, reason = _metaspot_cache_matches(
                    dgi_cache,
                    adata1,
                    required_obsm=(METASPOT_DGI_KEY,),
                    required_obsp=(METASPOT_GRAPH_KEY,),
                )
                dgi_meta = (
                    dgi_cache.uns.get("dgi_embeddings", {}).get(
                        METASPOT_DGI_KEY,
                        {},
                    )
                )
                valid = bool(
                    valid
                    and dgi_meta.get("node_unit") == "metaspot"
                    and int(dgi_meta.get("schema_version", -1))
                    == METASPOT_DGI_SCHEMA_VERSION
                    and dgi_meta.get("node_id_checksum")
                    == _obs_names_checksum(adata1.obs_names)
                    and dgi_meta.get("training_mode")
                    == execution_plan["dgi_training_mode"]
                )
                if not valid:
                    _logging.getLogger("cellscope").warning(
                        "Rejecting MetaSpot DGI cache: %s",
                        reason,
                    )
                    dgi_cache = None
            if dgi_cache is not None:
                if progress_callback:
                    progress_callback("Resume: load Module 8 MetaSpot DGI")
                adata1 = dgi_cache
            else:
                if progress_callback:
                    progress_callback("Module 8 · MetaSpot DGI Embedding")
                feature_keys = tuple(
                    adata1.uns["metaspot_graph"]["feature_keys"]
                )
                _, embedded = run_module8_dgi(
                    adata1,
                    adata1,
                    show_internal_progress=show_internal_progress,
                    dgi_epochs=dgi_epochs,
                    enable_dgi_sage=enable_dgi_sage,
                    out_dir=out_dir,
                    progress_callback=progress_callback,
                    graph_key=METASPOT_GRAPH_KEY,
                    feature_keys=feature_keys,
                    output_key=METASPOT_DGI_KEY,
                    sage_output_key=METASPOT_DGI_SAGE_KEY,
                    broadcast_to_adata1=False,
                    compatibility_alias=True,
                )
                adata1 = embedded
                _save_adata(adata1, out_dir, "module8_metaspot_adata1")
                _m8_metaspot_recomputed = True
        else:
            if progress_callback:
                progress_callback("Module 8 · MetaSpot DGI Embedding")
            feature_keys = tuple(adata1.uns["metaspot_graph"]["feature_keys"])
            _, embedded = run_module8_dgi(
                adata1,
                adata1,
                show_internal_progress=show_internal_progress,
                dgi_epochs=dgi_epochs,
                enable_dgi_sage=enable_dgi_sage,
                out_dir=out_dir,
                progress_callback=progress_callback,
                graph_key=METASPOT_GRAPH_KEY,
                feature_keys=feature_keys,
                output_key=METASPOT_DGI_KEY,
                sage_output_key=METASPOT_DGI_SAGE_KEY,
                broadcast_to_adata1=False,
                compatibility_alias=True,
            )
            adata1 = embedded
            _save_adata(adata1, out_dir, "module8_metaspot_adata1")
            _m8_metaspot_recomputed = True
        if _m8_metaspot_recomputed:
            _cache_chain_valid = False
        _logging.getLogger("cellscope").info(
            "Module 8 MetaSpot DGI ready: shape=%s",
            tuple(adata1.obsm[METASPOT_DGI_KEY].shape),
        )

    # Module 5: meta-domain clustering (allow resume)
    _m5_recomputed = False
    if _mod_lt('m5', start_mod) and _cache_chain_valid:
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module5_adata1.h5ad"))
        pd_cache = _load_dataframe_checkpoint(_intermediate_dir(out_dir), "module5_point_df")
        if a1_cache is not None and pd_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 5 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 5 cache (module5_adata1.h5ad, module5_point_df)")
            except Exception:
                pass
            adata1 = a1_cache
            point_df = pd_cache
            diag_simple = None
        else:
            if progress_callback: progress_callback("Module 5 · Meta-domain Clustering")
            adata1, diag_simple = run_module5_simple(
                adata1,
                point_df,
                speed_preset="ultra",
                use_spatial_feats=True,
                use_decell=True,
                use_cosine=True,
                enable_cap=True,
                show_internal_progress=show_internal_progress,
                progress_callback=progress_callback,
                module5_config_override={
                    "large_scale_method": (
                        execution_plan["metadomain_method"]
                        if execution_plan is not None
                        else "auto"
                    )
                },
            )
            _m5_recomputed = True
    else:
        if progress_callback: progress_callback("Module 5 · Meta-domain Clustering")
        adata1, diag_simple = run_module5_simple(
            adata1,
            point_df,
            speed_preset="ultra",
            use_spatial_feats=True,
            use_decell=True,
            use_cosine=True,
            enable_cap=True,
            show_internal_progress=show_internal_progress,
            progress_callback=progress_callback,
            module5_config_override={
                "large_scale_method": (
                    execution_plan["metadomain_method"]
                    if execution_plan is not None
                    else "auto"
                )
            },
        )
        _m5_recomputed = True
    if _m5_recomputed:
        _cache_chain_valid = False
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
    if "meta-domain" in point_df.columns:
        point_df = point_df.drop(columns=["meta-domain"])  # ensure fresh
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
        _logging.getLogger("cellscope").info("Module 5 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m5_recomputed) else ("recomputed" if _m5_recomputed else "cache")
            progress_callback("Module 5 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    if _io_should_write_figures():
        try:
            _plot_metadomain_k_selection_diagnostics(
                adata1,
                out_dir or os.getcwd(),
            )
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "MetaDomain clustering-diagnostic plotting failed: %s",
                exc,
            )

    # Module 6: Build adata2 from adata1 + meta-domains
    # Module 6: adata2 (allow resume)
    _m6_recomputed = False
    if _mod_lt('m6', start_mod) and _cache_chain_valid:
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module6_adata2.h5ad"))
        if a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 6 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 6 cache (module6_adata2.h5ad)")
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
        _logging.getLogger("cellscope").info("Module 6 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m6_recomputed) else ("recomputed" if _m6_recomputed else "cache")
            progress_callback("Module 6 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    if graph_unit == "metaspot":
        # Modules 7/8 already ran on adata1 before MetaDomain construction.
        # adata2 now carries pooled MetaSpot representations for annotation
        # Leiden; do not train a second DGI in the first implementation.
        final_csv = point_df.copy()
        return adata1, adata2, final_csv

    # Module 7: Graph construction on adata2 (weighted edges, optional same-cell downweight)
    # Module 7: Graph Construction (allow resume)
    _m7_recomputed = False
    if _mod_lt('m7', start_mod) and _cache_chain_valid:
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module7_adata2.h5ad"))
        if a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 7 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 7 cache (module7_adata2.h5ad)")
            except Exception:
                pass
            adata2 = a2_cache
        else:
            if progress_callback: progress_callback("Module 7 · Graph Construction")
            adata2 = build_module7_graph(
                adata2,
                show_internal_progress=show_internal_progress,
                scvi_epochs=scvi_epochs,
                out_dir=out_dir,
                progress_callback=progress_callback,
            )
            try:
                _save_adata(adata2, out_dir, "module7_adata2")
            except Exception:
                pass
            _m7_recomputed = True
    else:
        if progress_callback: progress_callback("Module 7 · Graph Construction")
        adata2 = build_module7_graph(
            adata2,
            show_internal_progress=show_internal_progress,
            scvi_epochs=scvi_epochs,
            out_dir=out_dir,
            progress_callback=progress_callback,
        )
        try:
            _save_adata(adata2, out_dir, "module7_adata2")
        except Exception:
            pass
        _m7_recomputed = True
    if _m7_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m7_recomputed) else ("recomputed" if _m7_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 7 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m7_recomputed) else ("recomputed" if _m7_recomputed else "cache")
            progress_callback("Module 7 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass
    # NOTE: single logging/progress_callback above is sufficient; avoid duplicate messages.

    # Module 8: Train DGI (GCN/SAGE) and backfill embeddings to adata2/adata1
    # Module 8: DGI embeddings (allow resume)
    _m8_recomputed = False
    if _mod_lt('m8', start_mod) and _cache_chain_valid:
        a1_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module8_adata1.h5ad"))
        a2_cache = _load_adata_checkpoint(os.path.join(_intermediate_dir(out_dir), "module8_adata2.h5ad"))
        if a1_cache is not None and a2_cache is not None:
            if progress_callback: progress_callback("Resume: load Module 8 cache")
            try:
                import logging as _logging
                _logging.getLogger("cellscope").info("Resume: using Module 8 cache (module8_adata1.h5ad, module8_adata2.h5ad)")
            except Exception:
                pass
            adata1, adata2 = a1_cache, a2_cache
        else:
            if progress_callback: progress_callback("Module 8 · Graph Embeddings (DGI)")
            adata1, adata2 = run_module8_dgi(
                adata1,
                adata2,
                show_internal_progress=show_internal_progress,
                dgi_epochs=dgi_epochs,
                enable_dgi_sage=enable_dgi_sage,
                out_dir=out_dir,
                progress_callback=progress_callback,
            )
            try:
                _save_adata(adata1, out_dir, "module8_adata1")
                _save_adata(adata2, out_dir, "module8_adata2")
            except Exception:
                pass
            _m8_recomputed = True
    else:
        if progress_callback: progress_callback("Module 8 · Graph Embeddings (DGI)")
        adata1, adata2 = run_module8_dgi(
            adata1,
            adata2,
            show_internal_progress=show_internal_progress,
            dgi_epochs=dgi_epochs,
            enable_dgi_sage=enable_dgi_sage,
            out_dir=out_dir,
            progress_callback=progress_callback,
        )
        try:
            _save_adata(adata1, out_dir, "module8_adata1")
            _save_adata(adata2, out_dir, "module8_adata2")
        except Exception:
            pass
        _m8_recomputed = True
    if _m8_recomputed:
        _cache_chain_valid = False
    try:
        import logging as _logging
        _suf = None if (_first_run and _m8_recomputed) else ("recomputed" if _m8_recomputed else "cache")
        _logging.getLogger("cellscope").info("Module 8 · Done%s", (f" ({_suf})" if _suf else ""))
    except Exception:
        pass
    if progress_callback:
        try:
            _suf = None if (_first_run and _m8_recomputed) else ("recomputed" if _m8_recomputed else "cache")
            progress_callback("Module 8 · Done" + (f" ({_suf})" if _suf else ""))
        except Exception:
            pass

    final_csv = point_df.copy()
    return adata1, adata2, final_csv


def _checkpoint_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f".{name}.parquet")

# Global I/O switches for intermediates, set by run_pipeline and/or params.yaml
_WRITE_INTERMEDIATE: Optional[bool] = None
_LITE_INTERMEDIATE: Optional[bool] = None
_WRITE_FIGURES: Optional[bool] = None

def _io_should_write_intermediate() -> bool:
    """Decide whether to write intermediate artifacts.

    Priority:
      1) explicit global override set by run_pipeline
      2) io.write_intermediate (if present in params.yaml)
      3) io.keep_intermediate (legacy semantics; False disables writing)
      4) default True
    """
    global _WRITE_INTERMEDIATE
    if _WRITE_INTERMEDIATE is not None:
        return bool(_WRITE_INTERMEDIATE)
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if 'write_intermediate' in io_cfg:
            return bool(io_cfg.get('write_intermediate'))
        if 'keep_intermediate' in io_cfg:
            return bool(io_cfg.get('keep_intermediate'))
    except Exception:
        pass
    return True

def _io_should_write_figures() -> bool:
    """Decide whether to write figures (PNG).

    Priority:
      1) explicit global override set by run_pipeline
      2) io.write_intermediate_figures (params.yaml)
      3) default True
    Note: figures are allowed even if other intermediates are disabled.
    """
    global _WRITE_FIGURES
    if _WRITE_FIGURES is not None:
        return bool(_WRITE_FIGURES)
    allow = True
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if 'write_intermediate_figures' in io_cfg:
            allow = bool(io_cfg.get('write_intermediate_figures'))
    except Exception:
        pass
    return allow


def _intermediate_dirname() -> str:
    """Return the configured intermediate folder name without creating it."""
    try:
        _cfg = load_params_yaml()
        dname = (_cfg.get('io', {}) if _cfg else {}).get('intermediate_dirname', 'intermediate')
        dname = str(dname).strip() or 'intermediate'
    except Exception:
        dname = 'intermediate'
    return dname


def _intermediate_dir(out_dir: str) -> str:
    """Return and create the configured intermediate output directory."""
    p = os.path.join(out_dir, _intermediate_dirname())
    os.makedirs(p, exist_ok=True)
    return p

def _atomic_dataframe_write(
    df: pd.DataFrame,
    path: str,
    *,
    file_format: str,
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=f".tmp.{file_format}",
        dir=directory,
    )
    os.close(fd)
    try:
        if file_format == "csv":
            df.to_csv(temporary, index=False)
        elif file_format == "parquet":
            df.to_parquet(temporary, index=False)
        else:
            raise ValueError(f"Unsupported dataframe format: {file_format}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _quarantine_stale_file(path: str) -> Optional[str]:
    """Move an older output aside so it cannot masquerade as this run."""
    if not os.path.isfile(path):
        return None
    candidate = f"{path}.stale"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path}.stale.{suffix}"
        suffix += 1
    os.replace(path, candidate)
    return candidate


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
        cell_codes, _ = pd.factorize(df["cell_id"], sort=False)
        partition_ids = np.mod(cell_codes, n_parts).astype(np.int32)
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
            frame.to_parquet(path, index=False)
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


_DEPRECATED_OUTPUT_ARTIFACTS = (
    "rapx_umap_cluster_annotation.png",
    "annotation_umap_leiden.png",
    "fine_annotation_proportions.png",
    "fine_annotation_proportions.csv",
    "fine_annotation_proportions.parquet",
)


def _remove_deprecated_output_artifacts(out_dir: str) -> List[str]:
    """Remove superseded files that would misrepresent a resumed run.

    These paths are deliberately exact and confined to the configured
    intermediate directory.  This prevents stale figures from surviving when
    a newer code version no longer emits them.
    """
    intermediate = os.path.join(out_dir, _intermediate_dirname())
    if not os.path.isdir(intermediate):
        return []
    removed: List[str] = []
    for name in _DEPRECATED_OUTPUT_ARTIFACTS:
        path = os.path.join(intermediate, name)
        if not os.path.isfile(path):
            continue
        try:
            os.unlink(path)
            removed.append(path)
            _logging.getLogger("cellscope").info(
                "Removed deprecated output artifact: %s",
                path,
            )
        except OSError as exc:
            _logging.getLogger("cellscope").warning(
                "Could not remove deprecated output artifact %s: %s",
                path,
                exc,
            )
    return removed


def _remove_inactive_method_artifacts(
    out_dir: str,
    config: Mapping[str, Any],
) -> List[str]:
    """Remove exact QC files emitted only by the inactive model path.

    Resumed runs intentionally share one output directory. Without this
    cleanup, a CA-DMGI/consensus-Leiden run can appear beside stale
    standard-DGI/KMeans figures and incorrectly imply that both generated the
    final result.
    """
    module5 = config.get("module5", {}) if isinstance(config, Mapping) else {}
    module8 = config.get("module8", {}) if isinstance(config, Mapping) else {}
    clustering_method = str(
        module5.get("clustering_method", "minibatch_kmeans")
        if isinstance(module5, Mapping)
        else "minibatch_kmeans"
    ).strip().lower().replace("-", "_")
    framework = str(
        module8.get("dgi_framework", "standard_dgi")
        if isinstance(module8, Mapping)
        else "standard_dgi"
    ).strip().lower().replace("-", "_")
    inactive_names: List[str] = []
    if clustering_method in {"consensus_leiden", "stability_leiden", "leiden"}:
        inactive_names.extend(
            [
                "metadomain_k_selection_diagnostics.png",
                "module5_k_selection_diagnostics.csv",
                "module5_k_selection_diagnostics.parquet",
            ]
        )
    else:
        inactive_names.extend(
            [
                "metadomain_consensus_leiden_diagnostics.png",
                "module5_consensus_leiden_diagnostics.csv",
                "module5_consensus_leiden_diagnostics.parquet",
            ]
        )
    legacy_framework = "ch_dmgi"  # legacy compatibility only
    if framework in {"cell_aware_dmgi", "ca_dmgi", legacy_framework, "dmgi"}:
        inactive_names.append("module8_dgi_GCN_curves.png")
    else:
        inactive_names.extend(
            [
                "module8_ca_dmgi_training.png",
                "module8_ca_dmgi_training_history.csv",
                "module8_ca_dmgi_training_history.parquet",
                "module8_ca_dmgi_summary.csv",
                "module8_ca_dmgi_summary.parquet",
            ]
        )
    inactive_names.extend(
        [
            "module8_ch_dmgi_training.png",  # legacy compatibility only
            "module8_ch_dmgi_training_history.csv",  # legacy compatibility only
            "module8_ch_dmgi_training_history.parquet",  # legacy compatibility only
            "module8_ch_dmgi_summary.csv",  # legacy compatibility only
            "module8_ch_dmgi_summary.parquet",  # legacy compatibility only
        ]
    )

    intermediate = os.path.join(out_dir, _intermediate_dirname())
    removed: List[str] = []
    for name in inactive_names:
        path = os.path.join(intermediate, name)
        if not os.path.isfile(path):
            continue
        try:
            os.unlink(path)
            removed.append(path)
            _logging.getLogger("cellscope").info(
                "Removed inactive method artifact: %s",
                path,
            )
        except OSError as exc:
            _logging.getLogger("cellscope").warning(
                "Could not remove inactive method artifact %s: %s",
                path,
                exc,
            )
    return removed


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

- `subcellular_annotation`: the final label. Inside each cell every one of the
  seven structures is placed on its best-scoring unit, so nearly every cell
  shows all seven. It is not gated by validation.
- `subcellular_annotation_validated`: only structures that passed the built-in
  held-out marker validation in this dataset; everything else is `Nucleus` or
  `Cytosol`. Use it for claims that a structure is really there.
- `subcellular_annotation_exploratory`: the validated view with a relaxed
  threshold.
- `subcellular_score__*` / `subcellular_q__*`: per-structure score and q-value
  of each MetaSpot.
- `evidence_annotation_report.json`: per-structure held-out enrichment with
  confidence interval, the level used, and whether it is supported.
  `unsupported_structures` lists the structures whose final labels are a best
  guess from the reference rather than a validated call.

## Clustering and method diagnostics (when figure output is enabled)

- `{intermediate_name}/metadomain_consensus_leiden_diagnostics.png`: default
  multi-resolution, multi-seed clustering stability/occupancy audit. The
  companion `module5_consensus_leiden_diagnostics.*` contains exact values.
- `{intermediate_name}/module8_ca_dmgi_training.png`: default Cell-Aware Deep
  Multiplex Graph Infomax (CA-DMGI) relation-specific expression/spatial losses
  and learned consensus-gate trajectory. The companion
  `module8_ca_dmgi_training_history.*` and `module8_ca_dmgi_summary.*` contain
  exact values and graph semantics.
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


def _save_dataframe(
    df: pd.DataFrame,
    out_dir: str,
    name: str,
    *,
    diagnostic: bool = False,
    force_parquet: bool = False,
):
    """Atomically save an intermediate or compact diagnostic table."""
    if not diagnostic and not _io_should_write_intermediate():
        return
    base = _intermediate_dir(out_dir)
    write_csv = True
    write_pq = True
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if diagnostic:
            if not bool(io_cfg.get("write_diagnostic_tables", True)):
                return
            write_csv = bool(io_cfg.get("write_diagnostic_csv", True))
            write_pq = bool(io_cfg.get("write_diagnostic_parquet", False))
        else:
            write_csv = bool(io_cfg.get('write_intermediate_csv', True))
            write_pq = bool(io_cfg.get('write_intermediate_parquet', True))
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not read I/O settings for table %s: %s; using defaults",
            name,
            exc,
        )
    if force_parquet:
        write_pq = True
    if not write_csv and not write_pq:
        _logging.getLogger("cellscope").debug(
            "Intermediate table disabled by I/O config: %s",
            name,
        )
        return
    _logging.getLogger("cellscope").info(
        "Saving intermediate table: %s",
        name,
    )
    written = []
    if write_csv:
        csv_path = os.path.join(base, f"{name}.csv")
        try:
            _atomic_dataframe_write(
                df,
                csv_path,
                file_format="csv",
            )
            written.append(csv_path)
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Failed to save intermediate CSV %s: %s",
                csv_path,
                exc,
            )
    if write_pq:
        try:
            pq_path = os.path.join(base, f"{name}.parquet")
            _atomic_dataframe_write(
                df,
                pq_path,
                file_format="parquet",
            )
            written.append(pq_path)
        except Exception as exc:
            _logging.getLogger("cellscope").warning(
                "Failed to save intermediate Parquet %s: %s",
                pq_path,
                exc,
            )
    if written:
        _logging.getLogger("cellscope").info(
            "Saved %s table | name=%s | rows=%d | columns=%d | files=%s",
            "diagnostic" if diagnostic else "intermediate",
            name,
            len(df),
            int(df.shape[1]),
            ", ".join(written),
        )

def _preferred_n_jobs() -> int:
    """Return preferred parallel worker count.

    Env vars (checked in order):
      CELLSCOPE_N_JOBS         -> explicit override
      CELLSCOP3E_N_JOBS        -> (legacy typo) still honored

    params.yaml (runtime.n_jobs) is honored when env vars are absent.

    Fallback heuristic: use ~1/3 of available CPUs, cap <=32 (legacy behavior).
    """
    for k in ("CELLSCOPE_N_JOBS", "CELLSCOP3E_N_JOBS"):
        try:
            v_raw = os.environ.get(k, "").strip()
            if v_raw:
                v = int(v_raw)
                if v > 0:
                    return v
        except Exception:
            pass
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        runtime_cfg = cfg.get("runtime", {}) or {}
        v = runtime_cfg.get("n_jobs")
        v_int = _coerce_pos_int(v)
        if v_int:
            return v_int
    except Exception:
        pass
    try:
        import multiprocessing
        cpu = max(1, multiprocessing.cpu_count())
    except Exception:
        cpu = 4
    # Default: use roughly half of available CPUs, capped at 32
    return max(1, min(32, cpu // 2 if cpu >= 2 else 1))

def _joblib_backend() -> str:
    # Default backend: 'loky' for maximum throughput on CPU-bound tasks.
    # Users can override via CELLSCOPE_JOBLIB_BACKEND env or runtime.joblib_backend.
    env_backend = os.environ.get("CELLSCOPE_JOBLIB_BACKEND")
    if env_backend:
        return env_backend
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        runtime_cfg = cfg.get("runtime", {}) or {}
        rt_backend = runtime_cfg.get("joblib_backend")
        if isinstance(rt_backend, str) and rt_backend.strip():
            return rt_backend.strip()
    except Exception:
        pass
    return "loky"

def _cluster_chunk_size(configured: Optional[int] = None) -> int:
    """Chunk size for per-cell parallel loops.

    Env: CELLSCOPE_CLUSTER_CHUNK (int)
    Config: module3.parallel_chunk_cells (with module2's legacy setting as a
    fallback when called outside the pipeline).
    Smaller chunks reduce peak memory (fewer simultaneous process payloads)
    at the cost of higher scheduler overhead.
    """
    try:
        env_v = os.environ.get("CELLSCOPE_CLUSTER_CHUNK", "").strip()
        if env_v:
            return max(16, min(1024, int(env_v)))
    except Exception:
        pass
    try:
        if configured is not None:
            return max(16, min(4096, int(configured)))
    except Exception:
        pass
    try:
        from .config import load_params_yaml
        cfg = load_params_yaml() or {}
        m3 = cfg.get("module3", {}) or {}
        cfg_v = m3.get("parallel_chunk_cells")
        if cfg_v is not None:
            return max(16, min(4096, int(cfg_v)))
        m2 = cfg.get("module2", {}) or {}
        cfg_v = m2.get("parallel_chunk_cells")
        if cfg_v is not None:
            return max(16, min(4096, int(cfg_v)))
    except Exception:
        pass
    try:
        v = int(os.environ.get("CELLSCOPE_CLUSTER_CHUNK", "512"))
        return max(128, min(4096, v))
    except Exception:
        return 256


def _state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "pipeline_state.json")

def _load_state(out_dir: str) -> Optional[dict]:
    try:
        import json
        p = _state_path(out_dir)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not read pipeline state from %s: %s",
            _state_path(out_dir),
            exc,
        )
    return None

def _save_state(out_dir: str, last_completed: str, extra: Optional[dict] = None) -> None:
    try:
        import time
        state = {"last_completed": last_completed, "ts": time.time()}
        if extra:
            state.update(extra)
        atomic_write_json(_state_path(out_dir), state)
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Could not persist pipeline state after stage %s: %s",
            last_completed,
            exc,
            exc_info=True,
        )

def _sanitize_obs_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    """Normalize obs values unsupported by older AnnData/HDF5 writers."""
    for col in adata.obs.columns:
        series = adata.obs[col]
        # AnnData/HDF5 natively supports categorical and nullable string
        # columns. Expanding them to Python objects used to perform one Python
        # call per value and destroyed compact category codes on large runs.
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_string_dtype(series.dtype) and series.dtype != object:
            continue
        if series.dtype != object:
            continue

        non_missing = series[series.notna()]
        sample = non_missing.iloc[:1024].to_numpy(dtype=object, copy=False)
        contains_sequence = any(
            isinstance(value, (list, tuple)) for value in sample
        )
        all_strings = all(isinstance(value, str) for value in sample)
        if contains_sequence:
            normalized = series.map(
                lambda value: ",".join(map(str, value))
                if isinstance(value, (list, tuple))
                else ("" if pd.isna(value) else str(value))
            )
            adata.obs[col] = normalized.astype(object)
        elif all_strings:
            if series.isna().any():
                adata.obs[col] = series.where(series.notna(), "")
        else:
            adata.obs[col] = series.where(series.notna(), "").astype(str)
    return adata


def _atomic_write_adata(
    adata: ad.AnnData,
    path: str,
    *,
    compression: str,
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp.h5ad",
        dir=directory,
    )
    os.close(fd)
    try:
        adata.write_h5ad(temporary, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _compact_final_adata1_for_write(
    adata1: ad.AnnData,
    *,
    enabled: bool,
):
    """Temporarily retain only the canonical MetaSpot DGI representation.

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
        if key == METASPOT_DGI_KEY
    ]
    marker_key = "cellscope_final_output_compaction"
    sentinel = object()
    original_marker = adata1.uns.get(marker_key, sentinel)
    adata1.obsm.clear()
    adata1.obsm.update(kept_obsm)
    adata1.obsp.clear()
    adata1.uns[marker_key] = {
        "enabled": True,
        "kept_obsm": [key for key, _ in kept_obsm],
        "omitted_obsm": [
            key for key, _ in original_obsm if key != METASPOT_DGI_KEY
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


def _save_adata(adata: ad.AnnData, out_dir: str, name: str, compression: str = "lzf"):
    if not _io_should_write_intermediate():
        return
    tmp_path = None
    original_obs = adata.obs
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.h5ad")
        tmp_path = f"{p}.tmp-{os.getpid()}"
        _logging.getLogger("cellscope").info(
            "Saving intermediate AnnData: %s",
            p,
        )
        # Avoid mutating the in-memory schema merely for writer compatibility.
        adata.obs = original_obs.copy(deep=False)
        _sanitize_obs_for_h5ad(adata)
        try:
            import anndata as _ad
            _ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        adata.write_h5ad(tmp_path, compression=compression)
        os.replace(tmp_path, p)
        tmp_path = None
        _logging.getLogger("cellscope").info(
            "Saved intermediate AnnData: %s",
            p,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Failed to save intermediate AnnData %s: %s",
            name,
            exc,
            exc_info=True,
        )
    finally:
        adata.obs = original_obs
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _save_fig(fig, out_dir: str, name: str, *, dpi: int = 150):
    if not _io_should_write_figures():
        return
    temporary = None
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.png")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp.png",
            dir=os.path.dirname(p),
        )
        os.close(fd)
        fig.savefig(temporary, bbox_inches='tight', dpi=int(dpi))
        os.replace(temporary, p)
        temporary = None
        _logging.getLogger("cellscope").info(
            "Saved diagnostic figure | name=%s | path=%s",
            name,
            p,
        )
    except Exception as exc:
        _logging.getLogger("cellscope").warning(
            "Failed to save diagnostic figure %s: %s",
            name,
            exc,
            exc_info=True,
        )
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _plot_metadomain_k_selection_diagnostics(
    adata1: ad.AnnData,
    out_dir: str,
) -> Optional[pd.DataFrame]:
    """Save the sampled automatic-K evidence alongside Module 5 outputs."""
    metadata = adata1.uns.get("module5_metadomain_clustering", {})
    if metadata.get("clustering_method") == "consensus_leiden":
        rows: List[Dict[str, Any]] = []
        selected_by_block: Dict[str, float] = {}
        for block, block_meta in dict(
            metadata.get("consensus_by_block", {})
        ).items():
            resolutions = np.asarray(
                block_meta.get("candidate_resolution", []), dtype=float
            )
            stability = np.asarray(
                block_meta.get("candidate_stability_ari", []), dtype=float
            )
            n_clusters = np.asarray(
                block_meta.get("candidate_n_clusters", []), dtype=int
            )
            average_size = np.asarray(
                block_meta.get("candidate_average_size", []), dtype=float
            )
            small_fraction = np.asarray(
                block_meta.get("candidate_small_cluster_fraction", []),
                dtype=float,
            )
            nonrecurrent_fraction = np.asarray(
                block_meta.get("candidate_nonrecurrent_fraction", []),
                dtype=float,
            )
            feasible = np.asarray(
                block_meta.get("candidate_feasible", []), dtype=bool
            )
            if not (
                resolutions.size
                and all(
                    values.size == resolutions.size
                    for values in (
                        stability,
                        n_clusters,
                        average_size,
                        small_fraction,
                        nonrecurrent_fraction,
                        feasible,
                    )
                )
            ):
                continue
            selected_resolution = float(
                block_meta.get("selected_resolution", resolutions[0])
            )
            selected_by_block[str(block)] = selected_resolution
            for index, resolution in enumerate(resolutions):
                rows.append(
                    {
                        "block": str(block),
                        "resolution": float(resolution),
                        "stability_ari": float(stability[index]),
                        "n_clusters": int(n_clusters[index]),
                        "average_metaspots_per_cluster": float(
                            average_size[index]
                        ),
                        "small_cluster_fraction": float(
                            small_fraction[index]
                        ),
                        "nonrecurrent_fraction": float(
                            nonrecurrent_fraction[index]
                        ),
                        "feasible": bool(feasible[index]),
                        "selected": bool(
                            np.isclose(resolution, selected_resolution)
                        ),
                    }
                )
        diagnostics = pd.DataFrame(rows)
        if diagnostics.empty:
            return None
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 4.6), constrained_layout=True
        )
        palette = {
            "N": _SUBCELLULAR_LABEL_PALETTE["Nucleus"],
            "C": _SUBCELLULAR_LABEL_PALETTE["Cytosol"],
            "A": _SUBCELLULAR_LABEL_PALETTE["Unknown"],
        }
        for block, group in diagnostics.groupby("block", sort=False):
            color = palette.get(str(block), "#777777")
            axes[0].plot(
                group["resolution"],
                group["stability_ari"],
                marker="o",
                color=color,
                label=str(block),
            )
            axes[1].plot(
                group["resolution"],
                group["n_clusters"],
                marker="o",
                color=color,
                label=str(block),
            )
            axes[2].plot(
                group["resolution"],
                100.0 * group["nonrecurrent_fraction"],
                marker="o",
                color=color,
                label=f"{block}: nonrecurrent",
            )
            axes[2].plot(
                group["resolution"],
                100.0 * group["small_cluster_fraction"],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{block}: small",
            )
            selected = selected_by_block[str(block)]
            for axis in axes:
                axis.axvline(selected, color=color, linestyle=":", alpha=0.7)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("Cross-seed adjusted Rand index")
        axes[0].set_title("Assignment stability")
        axes[1].set_ylabel("Candidate MetaDomains")
        axes[1].set_title("Resolution and granularity")
        axes[2].set_ylabel("MetaSpots (%)")
        axes[2].set_title("Occupancy and cross-cell recurrence")
        for axis in axes:
            axis.set_xlabel("Leiden resolution")
            axis.legend(frameon=False, fontsize="small")
        fig.suptitle(
            "Stability-selected consensus Leiden",
            fontsize=16,
            fontweight="bold",
        )
        _save_fig(fig, out_dir, "metadomain_consensus_leiden_diagnostics")
        plt.close(fig)
        _save_dataframe(
            diagnostics,
            out_dir,
            "module5_consensus_leiden_diagnostics",
            diagnostic=True,
        )
        return diagnostics

    selection = metadata.get("k_selection", {})
    candidate_k = np.asarray(selection.get("candidate_k", []))
    distortion = np.asarray(selection.get("distortion", []), dtype=float)
    if candidate_k.size < 2 or distortion.size != candidate_k.size:
        return None

    def _aligned(name: str, default: float = np.nan) -> np.ndarray:
        values = np.asarray(selection.get(name, []), dtype=float)
        if values.size == candidate_k.size:
            return values
        return np.full(candidate_k.size, default, dtype=float)

    diagnostics = pd.DataFrame(
        {
            "K": candidate_k.astype(int),
            "heldout_distortion": distortion,
            "stability_ari": _aligned("stability_ari"),
            "small_cluster_fraction": _aligned(
                "small_cluster_fraction"
            ),
            "empty_cluster_fraction": _aligned(
                "empty_cluster_fraction"
            ),
        }
    ).sort_values("K")
    selected_k = int(selection.get("selected_k", candidate_k[0]))
    knee_k = int(selection.get("knee_k", selected_k))
    feasible = np.asarray(selection.get("feasible", []), dtype=bool)
    if feasible.size == candidate_k.size:
        diagnostics["feasible"] = feasible
    else:
        diagnostics["feasible"] = True
    diagnostics["selected"] = diagnostics["K"].eq(selected_k)
    diagnostics["distortion_knee"] = diagnostics["K"].eq(knee_k)

    blue = "#4C78A8"
    orange = "#F28E2B"
    grey = "#9D9D9D"
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.6),
        constrained_layout=True,
    )

    axes[0].plot(
        diagnostics["K"],
        diagnostics["heldout_distortion"],
        marker="o",
        color=blue,
        linewidth=2,
    )
    axes[0].scatter(
        [selected_k],
        diagnostics.loc[
            diagnostics["selected"],
            "heldout_distortion",
        ],
        s=90,
        color=orange,
        edgecolor="black",
        zorder=4,
        label=f"Selected K={selected_k}",
    )
    axes[0].axvline(knee_k, color=grey, linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Held-out clustering distortion")
    axes[0].set_xlabel("Candidate K (log scale)")
    axes[0].set_ylabel(
        "Mean 1 − cosine similarity"
        if bool(selection.get("use_cosine", True))
        else "Mean squared distance"
    )
    axes[0].legend(frameon=False)

    axes[1].plot(
        diagnostics["K"],
        diagnostics["stability_ari"],
        marker="o",
        color=blue,
        linewidth=2,
    )
    stability_min = float(selection.get("stability_min", 0.4))
    axes[1].axhline(
        stability_min,
        color=grey,
        linestyle="--",
        label=f"Minimum={stability_min:.2f}",
    )
    axes[1].axvline(selected_k, color=orange, linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Cross-seed assignment stability")
    axes[1].set_xlabel("Candidate K (log scale)")
    axes[1].set_ylabel("Adjusted Rand index")
    axes[1].legend(frameon=False)

    axes[2].plot(
        diagnostics["K"],
        100.0 * diagnostics["small_cluster_fraction"],
        marker="o",
        color=blue,
        linewidth=2,
        label="Small-cluster MetaSpots",
    )
    axes[2].plot(
        diagnostics["K"],
        100.0 * diagnostics["empty_cluster_fraction"],
        marker="s",
        color=grey,
        linewidth=1.5,
        label="Empty clusters",
    )
    max_small = float(
        selection.get("max_small_cluster_fraction", 0.01)
    )
    axes[2].axhline(
        100.0 * max_small,
        color=grey,
        linestyle="--",
        label=f"Maximum={100.0 * max_small:.1f}%",
    )
    axes[2].axvline(selected_k, color=orange, linestyle=":")
    axes[2].set_xscale("log")
    axes[2].set_title("Candidate occupancy diagnostics")
    axes[2].set_xlabel("Candidate K (log scale)")
    axes[2].set_ylabel("Fraction (%)")
    axes[2].legend(frameon=False, fontsize="small")

    reason = str(selection.get("selection_reason", ""))
    fig.suptitle(
        "MetaDomain automatic K selection"
        f" · {reason.replace('_', ' ')}",
        fontsize=16,
        fontweight="bold",
    )
    _save_fig(fig, out_dir, "metadomain_k_selection_diagnostics")
    plt.close(fig)
    _save_dataframe(
        diagnostics,
        out_dir,
        "module5_k_selection_diagnostics",
        diagnostic=True,
    )
    return diagnostics


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
        "prepare_adata",
        "final_table",
        "write_outputs",
    ]] = None,
    show_internal_progress: bool = False,
    enable_annotation: Optional[bool] = None,
    annotation_method: Optional[str] = None,
    scvi_epochs: Optional[int] = None,
    dgi_epochs: Optional[int] = None,
    enable_dgi_sage: Optional[bool] = None,
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
        elif save_intermediate is None and 'keep_intermediate' in io_cfg:
            ki = bool(io_cfg.get('keep_intermediate'))
            save_intermediate = bool(ki)
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
        m7_cfg = cfg.get('module7', {}) if cfg else {}
        m8_cfg = cfg.get('module8', {}) if cfg else {}
        if scvi_epochs is None and isinstance(
            m7_cfg.get('scvi_epochs'),
            (int, float),
        ):
            scvi_epochs = int(m7_cfg.get('scvi_epochs'))
        if dgi_epochs is None and isinstance(
            m8_cfg.get('dgi_epochs'),
            (int, float),
        ):
            dgi_epochs = int(m8_cfg.get('dgi_epochs'))
        if enable_dgi_sage is None and 'enable_dgi_sage' in m8_cfg:
            enable_dgi_sage = bool(m8_cfg.get('enable_dgi_sage'))
    except Exception as exc:
        logger.warning("Could not resolve module runtime settings: %s", exc)
    scvi_epochs = int(100 if scvi_epochs is None else scvi_epochs)
    dgi_epochs = int(200 if dgi_epochs is None else dgi_epochs)
    enable_dgi_sage = bool(
        False if enable_dgi_sage is None else enable_dgi_sage
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
        os.environ.get("CELLSCOPE_CONFIG", "(package default)"),
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
    try:
        global _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES
        _WRITE_INTERMEDIATE = bool(save_intermediate)
        _LITE_INTERMEDIATE = bool(lite_intermediate)
        # Figures default to True unless explicitly disabled
        if 'write_intermediate_figures' in (io_cfg or {}):
            _WRITE_FIGURES = bool(io_cfg.get('write_intermediate_figures'))
        else:
            _WRITE_FIGURES = True
        # Respect annotation.save_plots if present
        try:
            ann_cfg = cfg.get('annotation', {}) if cfg else {}
            if 'save_plots' in ann_cfg and _WRITE_FIGURES is not None:
                _WRITE_FIGURES = _WRITE_FIGURES and bool(ann_cfg.get('save_plots'))
        except Exception:
            pass
    except Exception:
        pass
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
        legacy_map = {"prepare_adata": "build_anndata", "final_table": "annotate"}
        start_from = legacy_map.get(resume_from, resume_from)
        logger.info("Resume requested from %s", start_from)
    else:
        # Choose behavior based on resume_policy
        policy = str(resume_policy or "minimal").lower()
        if policy == "force":
            start_from = "load_inputs"
            logger.info("Resume policy=force; starting from %s", start_from)
        elif policy == "auto":
            # legacy: continue from last completed + 1 if available
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
    _remove_deprecated_output_artifacts(out_dir)
    _remove_inactive_method_artifacts(out_dir, cfg)

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
            if bool(module1_cfg.get("remove_foreign_transcripts", False)):
                from .spillover import relative_type_expression

                keep = relative_type_expression(spatial_df) > float(
                    module1_cfg.get("foreign_max_relative_expression", 0.2)
                )
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
                scvi_epochs=scvi_epochs,
                dgi_epochs=dgi_epochs,
                enable_dgi_sage=enable_dgi_sage,
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
    if bool(enable_annotation) and ann_method in {"rapx", "rap-x"}:
        raise ValueError(
            "annotation.method 'rapx' was removed; use 'evidence'. The last "
            "version with RAP-X is the git tag 'legacy-rapx'."
        )
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
    if run_annotation and resume_enabled and stage_to_idx["annotate"] < start_idx:
        annotation_cache = _resume_annotation_outputs(out_dir)
        annotation_cache_ready = bool(
            annotation_cache
            and annotation_cache[0] is not None
            and annotation_cache[2] is not None
        )
    if run_annotation and _should_run("annotate", annotation_cache_ready):
        if resume_enabled and stage_to_idx["annotate"] < start_idx and not annotation_cache_ready:
            logger.info("Resume requested skip for annotate but checkpoint missing → re-running")
        if progress_callback: progress_callback(f"Annotation ({ann_method})")
        try:
            adata1, final_data = _annotate_with_evidence(
                adata1, final_data, gold_data, out_dir
            )
            required_annotation_columns = {
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
    global _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES
    previous = (
        _WRITE_INTERMEDIATE,
        _LITE_INTERMEDIATE,
        _WRITE_FIGURES,
    )
    try:
        return _run_pipeline_impl(*args, **kwargs)
    finally:
        (
            _WRITE_INTERMEDIATE,
            _LITE_INTERMEDIATE,
            _WRITE_FIGURES,
        ) = previous


# ====== Optional K1/K2: Clustering + RNALocate/APEX-based annotation ======


# --- RNALocate L1 mapping and projection to 7 compartments ---
import re as _re

SEVEN_KEYS   = ['NUCLEUS','NUCLEOLUS','LAMINA','NUCLEAR_PORE','CYTOSOL','ER','OMM']
SEVEN_PRETTY = {'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
                'NUCLEAR_PORE':'Nuclear_Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM'}
SEVEN_COLS   = [SEVEN_PRETTY[k] for k in SEVEN_KEYS]

L1 = [
    'NUCLEUS','NUC_LEM','NUC_PORE','NUCLEOLUS','CYTOSOL','RNP_GRANULE','RIBOSOME_CYTO',
    'ER','GOLGI','ENDOLYSO','PEROXISOME','MITO_OMM','MITO_IMM','MITO_MATRIX','MITO_UNSPEC',
    'PM_JUNCTION','CYTO_CORTEX_POLARITY','NEURONAL','CELL_DIVISION','EV','RIBOSOME_ORGANELLAR','OTHER',
]

_MULTI_SEP = _re.compile(r'[;,/|]+')

def _norm_loc_txt(s: str) -> str:
    return _re.sub(r'\s+', ' ', str(s).strip().lower())

# Explicit mapping (subset from user-provided list; extended as needed)
L1_EXPLICIT = {
    'nucleolus': 'NUCLEOLUS',
    'nucleus(exclusion from nucleoli)': 'NUCLEUS', 'nucleus': 'NUCLEUS', 'nuclear ': 'NUCLEUS', 'nucleoplasm': 'NUCLEUS',
    'nuclear envelope': 'NUC_LEM', 'nuclear inner membrane': 'NUC_LEM', 'nuclear outer membrane': 'NUC_LEM', 'nuclear periphery': 'NUC_LEM',
    'nuclear lamina': 'NUC_LEM', 'nuclear membrane': 'NUC_LEM',
    'nuclear pore complex': 'NUC_PORE', 'nuclear pore': 'NUC_PORE', 'npc': 'NUC_PORE',
    'cytosol': 'CYTOSOL', 'cytoplasm': 'CYTOSOL', 'cytosolic': 'CYTOSOL', 'perinuclear ': 'CYTOSOL',
    'stress granule': 'RNP_GRANULE', 'p-body': 'RNP_GRANULE',
    'cytosolic ribosome': 'RIBOSOME_CYTO', 'ribosome': 'RIBOSOME_CYTO',
    'endoplasmic reticulum': 'ER', 'er lumen': 'ER', 'er membrane': 'ER', 'rough endoplasmic reticulum': 'ER',
    'golgi apparatus': 'GOLGI', 'endosome': 'ENDOLYSO', 'lysosome': 'ENDOLYSO', 'autophagosome': 'ENDOLYSO',
    'peroxisome': 'PEROXISOME',
    'mitochondrial outer membrane': 'MITO_OMM', 'outer mitochondrial membrane': 'MITO_OMM',
    'mitochondrial inner membrane': 'MITO_IMM', 'mitochondrial matrix': 'MITO_MATRIX',
    'mitochondrial ribosome': 'MITO_UNSPEC', 'mitochondrion': 'MITO_UNSPEC',
    'plasma membrane': 'PM_JUNCTION', 'tight junction': 'PM_JUNCTION', 'cell junction': 'PM_JUNCTION',
    'cell cortex': 'CYTO_CORTEX_POLARITY', 'lamellipodium': 'CYTO_CORTEX_POLARITY', 'pseudopodium': 'CYTO_CORTEX_POLARITY',
    'axon': 'NEURONAL', 'dendrite': 'NEURONAL', 'synapse': 'NEURONAL',
    'centrosome': 'CELL_DIVISION', 'spindle': 'CELL_DIVISION',
    'extracellular vesicle': 'EV', 'extracellular exosome': 'EV',
    'vesicle': 'OTHER', 'membrane': 'OTHER',
}

L1_REGEX = [
    ('NUCLEOLUS',     r'\bnucleol\w+'),
    ('NUC_LEM',       r'\bnuclear (envelope|lamina)\b|\binner nuclear membrane\b|\bouter nuclear membrane\b|\bnuclear periphery\b'),
    ('NUC_PORE',      r'\bnuclear pore( complex)?\b|\bNPC\b'),
    ('NUCLEUS',       r'\bnucleus\b|\bnuclear(?! (envelope|lamina|pore))\b|\bchromatin\b'),
    ('CYTOSOL',       r'\bcytosol(ic)?\b|\bcytoplasm(ic)?\b|\bperinuclear(?! (envelope|membrane))\b'),
    ('RNP_GRANULE',   r'\bstress granule\b|\bP-?body\b|\bRNP\b'),
    ('ER',            r'\b(endoplasmic reticulum|ER)\b|\ber lumen\b|\ber membrane\b'),
    ('MITO_OMM',      r'\bouter (mitochondrial|mitochondrion) membrane\b|\bmitochondrial outer membrane\b'),
    ('MITO_IMM',      r'\bmitochondrial inner membrane\b'),
    ('MITO_UNSPEC',   r'\bmitochondri(on|a)l? (?!inner|outer)\b|\bmitochondrion\b'),
    ('PM_JUNCTION',   r'\bplasma membrane\b|\b(cell )?junction\b|\btight junction\b'),
    ('CYTO_CORTEX_POLARITY', r'\bcortex\b|\bapical\b|\bbasal\b|\blamellipod\w+|\bpseudopod\w+|\bprotrusion\b'),
    ('NEURONAL',      r'\baxon\w*|\bdendrit\w*|\bsynaps\w*'),
    ('ENDOLYSO',      r'\bendosome\b|\blysosome\b|\bautophagosome\b'),
    ('EV',            r'\bextracellular (vesicle|exosome)\b|\b(apoptotic|micro|nano)vesicle\b'),
    ('CELL_DIVISION', r'\bcentrosome\b|\bspindle\b|\bmitotic apparatus\b'),
    ('PEROXISOME',    r'\bperoxisome\b'),
    ('GOLGI',         r'\bgolgi\b'),
]

def map_loc_to_l1(loc: str) -> str:
    if not isinstance(loc, str) or not loc.strip():
        return 'OTHER'
    s = _norm_loc_txt(loc)
    for key in sorted(L1_EXPLICIT.keys(), key=len, reverse=True):
        if key and key in s:
            return L1_EXPLICIT[key]
    for l1, patt in L1_REGEX:
        if _re.search(patt, s, flags=_re.IGNORECASE):
            return l1
    return 'OTHER'


def build_gold_from_rnalocate_l1(rna_locate_df: pd.DataFrame,
                                 species=("Homo sapiens","Mus musculus"),
                                 rna_types=("mRNA",),
                                 score_col='RNALocate_Score',
                                 agg_mode='prob_or',
                                 score_floor=0.5,
                                 use_pubmed_weight=True) -> pd.DataFrame:
    need = {'Species','RNA_Symbol','RNA_Type','Subcellular_Localization',score_col}
    miss = need - set(rna_locate_df.columns)
    if miss:
        raise ValueError(f"RNALocate is missing required columns: {miss}")
    df = rna_locate_df.copy()
    if species is not None:
        df = df[df['Species'].isin(species)]
    if rna_types is not None:
        df = df[df['RNA_Type'].isin(rna_types)]
    def _clean_symbol(x):
        if not isinstance(x, str): return None
        x = x.strip(); return x.upper() if x else None
    df['Common_Gene'] = df['RNA_Symbol'].map(_clean_symbol)
    df = df[df['Common_Gene'].notna()]
    df = df.assign(Subcellular_Localization=df['Subcellular_Localization'].astype(str))
    df = df.assign(_loc_split=df['Subcellular_Localization'].map(lambda s: [t.strip() for t in _MULTI_SEP.split(s) if t.strip()]))
    df = df.explode('_loc_split', ignore_index=True)
    df['L1'] = df['_loc_split'].map(map_loc_to_l1)
    df['score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0.0).clip(0,1)
    df = df[df['score'] >= float(score_floor)]
    if use_pubmed_weight and 'PubMed_ID' in df.columns:
        pm = df['PubMed_ID'].notna() & (df['PubMed_ID'].astype(str).str.strip()!='')
        df.loc[pm,'score'] = (df.loc[pm,'score'] + 0.05).clip(0,1)
    if agg_mode == 'prob_or':
        agg = df.groupby(['Common_Gene','L1'])['score'].apply(lambda s: 1.0 - np.prod(1.0 - s.values))
    elif agg_mode == 'max':
        agg = df.groupby(['Common_Gene','L1'])['score'].max()
    else:
        raise ValueError("'agg_mode' supports only 'prob_or' or 'max'.")
    tab = agg.unstack('L1').reindex(columns=L1).fillna(0.0)
    tab.index.name = 'Common_Gene'
    return tab


def project_l1_to_seven(l1_df: pd.DataFrame, mode='strict', mito_non_omm=None, fold_misc_to_cytosol=True) -> pd.DataFrame:
    def _get(col):
        return pd.to_numeric(l1_df.get(col, pd.Series(0.0, index=l1_df.index)), errors='coerce').fillna(0.0).astype(float)
    out = pd.DataFrame(index=l1_df.index, columns=SEVEN_COLS, dtype=float)
    out['Nucleus']       = _get('NUCLEUS')
    out['Nucleolus']     = _get('NUCLEOLUS')
    out['Lamina']        = _get('NUC_LEM')
    out['Nuclear_Pore']  = _get('NUC_PORE')
    cyt_base = _get('CYTOSOL') + _get('RNP_GRANULE') + _get('RIBOSOME_CYTO')
    if mode == 'relaxed' and fold_misc_to_cytosol:
        cyt_base = (cyt_base + _get('GOLGI') + _get('ENDOLYSO') + _get('PEROXISOME') + _get('PM_JUNCTION') + _get('CYTO_CORTEX_POLARITY') + _get('CELL_DIVISION'))
    out['Cytosol'] = np.clip(cyt_base, 0.0, 1.0)
    out['ER'] = _get('ER')
    omm = _get('MITO_OMM')
    if mode == 'relaxed' and mito_non_omm in ('OMM','Cytosol'):
        add_mito = _get('MITO_IMM') + _get('MITO_MATRIX') + _get('MITO_UNSPEC')
        if mito_non_omm == 'OMM':
            omm = 1.0 - (1.0 - omm) * (1.0 - add_mito)
        elif mito_non_omm == 'Cytosol':
            out['Cytosol'] = 1.0 - (1.0 - out['Cytosol']) * (1.0 - add_mito)
    out['OMM'] = np.clip(omm, 0.0, 1.0)
    out = out.fillna(0.0); out.index.name = 'Common_Gene'
    return out


def _dedup_columns_by(df: pd.DataFrame, how='prob_or'):
    if not df.columns.duplicated().any():
        return df
    X = df.T
    if how == 'prob_or':
        def por(x):
            x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float)
            return 1.0 - np.prod(1.0 - x.values)
        agg = X.groupby(level=0, sort=False).agg(por)
    elif how == 'max':
        agg = X.groupby(level=0, sort=False).max()
    else:
        raise ValueError("'how' supports only 'prob_or' or 'max'.")
    return agg.T


def _dedup_index_by(df: pd.DataFrame, how='prob_or'):
    idx_up = df.index.astype(str).str.upper()
    X = df.copy(); X.index = idx_up
    if not idx_up.duplicated().any():
        return X
    if how == 'prob_or':
        def por(x):
            x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float)
            return 1.0 - np.prod(1.0 - x.values)
        return X.groupby(level=0, sort=False).agg(por)
    elif how == 'max':
        return X.groupby(level=0, sort=False).max()
    else:
        raise ValueError("'how' supports only 'prob_or' or 'max'.")


def merge_gold(existing_gold_df: pd.DataFrame, new_gold_df: pd.DataFrame, fuse='max'):
    col_alias = {
        'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
        'NUCLEAR_PORE':'Nuclear_Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM',
        'ERM':'ER','ER_LUMEN':'ER','ER_Lumen':'ER','ER Membrane':'ER','ER membrane':'ER'
    }
    agg_how = 'prob_or' if fuse=='prob_or' else 'max'
    def _prep(df):
        X = df.copy()
        if 'Common_Gene' in X.columns and X.index.name != 'Common_Gene':
            X = X.set_index('Common_Gene')
        X.columns = pd.Index([str(c).strip() for c in X.columns])
        X = X.rename(columns=col_alias)
        if X.columns.duplicated().any():
            X = _dedup_columns_by(X, how=agg_how)
        keep = [c for c in X.columns if c in SEVEN_COLS]
        X = X[keep]
        for c in SEVEN_COLS:
            if c not in X.columns:
                X[c] = 0.0
        X = X.reindex(columns=SEVEN_COLS, copy=False)
        X = _dedup_index_by(X, how=agg_how)
        X = X.apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0).astype(float)
        assert X.columns.is_unique, "Column names are still not unique. Please check upstream data or alias mapping."
        return X
    A = _prep(existing_gold_df)
    B = _prep(new_gold_df)
    all_idx = A.index.union(B.index)
    A = A.reindex(all_idx).fillna(0.0)
    B = B.reindex(all_idx).fillna(0.0)
    if fuse == 'prob_or':
        out = 1.0 - (1.0 - A) * (1.0 - B)
    elif fuse == 'max':
        out = np.maximum(A, B)
    else:
        raise ValueError("'fuse' supports only 'max' or 'prob_or'.")
    out.index.name = 'Common_Gene'
    return out


def _load_manual_marker_data(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load optional manual marker table and normalize row-wise to 0-1.

    Expected columns: `Common_Gene` + region columns (Nucleus/Nucleolus/.../OMM).
    If `Common_Gene` absent but index is the gene names, the index will be reset.
    Missing file or parse errors return None (non-fatal).
    """
    if path is None:
        path = bundled_resource_path("manual_marker.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_table(path)
        except Exception:
            return None

    if 'Common_Gene' not in df.columns and df.index.name == 'Common_Gene':
        df = df.reset_index()
    if 'Common_Gene' not in df.columns:
        return None

    val_cols = [c for c in df.columns if c != 'Common_Gene']
    if len(val_cols) == 0:
        return None

    X = df[val_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
    X = X.subtract(X.min(axis=1), axis=0)
    span = (X.max(axis=1) - X.min(axis=1)).replace(0, 1.0)
    X = X.div(span, axis=0)
    out = pd.concat([df[['Common_Gene']].reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Drop all-zero rows to avoid inflating sets with null markers.
    try:
        mask_nonzero = (X.sum(axis=1) > 0)
        out = out.loc[mask_nonzero].reset_index(drop=True)
    except Exception:
        pass
    return out


# ---------- 目标区域与别名 ----------
TARGET_REGIONS = ['NUCLEUS','NUCLEOLUS','LAMINA','NUCLEAR_PORE','CYTOSOL','ER','OMM']
GOLD_ALIAS = {
    'NUCLEUS':'NUCLEUS','NUCLEOLUS':'NUCLEOLUS','LAMINA':'LAMINA',
    'NUCLEAR_PORE':'NUCLEAR_PORE','NUCLEARPORE':'NUCLEAR_PORE',
    'CYTOSOL':'CYTOSOL','CYTOSOLIC':'CYTOSOL',
    'ER':'ER','ERM':'ER','ER_LUMEN':'ER','ER_LUMINAL':'ER',
    'OMM':'OMM','MITOCHONDRIA_OMM':'OMM',
}
PRETTY = {'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
          'NUCLEAR_PORE':'Nuclear Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM'}
NUCLEAR_LABELS = frozenset({'Nucleus', 'Nucleolus', 'Lamina', 'Nuclear Pore'})
CYTOPLASMIC_LABELS = frozenset(
    {'Cytosol', 'ER', 'OMM', 'ER-OMM shared'}
)

# 可视化用的顺序与颜色
PRETTY_ORDER = ['Cytosol','ER','Lamina','Nuclear Pore','Nucleolus','Nucleus','OMM']
SIG_COLOR    = {
    '***': '#d62728',
    '**': '#ff7f0e',
    '*': '#2ca02c',
    'ns': '#7f7f7f',
    'NA': '#bdbdbd',
}

# ---------- 参数 ----------
TARGET_SET_SIZE = 600
MIN_SET_SIZE    = 200
MAX_SET_SIZE    = 1000
MIN_TOP_VALUE   = 0.1
USE_MARGIN      = True

TOPK_OVERLAP_FRAC = 0.3
TOPK_OVERLAP_MIN  = 250

FDR_CUTOFF      = 0.05
Z_CUTOFF        = 1
TIE_DELTA       = 0.35
ALWAYS_ASSIGN   = False

PROMOTE_TOP2    = True
DELTA_PROMOTE   = 1
REL_PROMOTE     = 0.9

# 新增：第4个通道权重（空间）
CHANNEL_WEIGHTS = (1, 1, 1, 0.25)  # MWU, Proj, Hyp, soft geometry

# ---------- Gold 基因大小写自适应 ----------
def harmonize_gold_gene_case(gold_df: pd.DataFrame, target_var_names: Optional[pd.Index] = None) -> pd.DataFrame:
    """Normalize `Common_Gene` casing and pick the style with better overlap to target genes.

    Mouse数据常用首字母大写+其余小写的写法（例如 Actb），而 Gold 表往往是全大写。
    这里同时计算两种写法与目标 adata.var_names 的重叠度，选择更优者并去重聚合。
    """
    gd = gold_df.copy()

    # Ensure Common_Gene is a column for processing.
    if 'Common_Gene' not in gd.columns and gd.index.name == 'Common_Gene':
        gd = gd.reset_index()
    if 'Common_Gene' not in gd.columns:
        return gd

    genes = gd['Common_Gene'].astype(str)
    upper = genes.str.upper()
    mouse_style = genes.str.capitalize()

    chosen = upper
    if target_var_names is not None:
        tgt = pd.Index(target_var_names.astype(str))
        tgt_up = tgt.str.upper()
        tgt_set = set(tgt)
        tgt_up_set = set(tgt_up)

        # Score exact case matches for mouse-style, and uppercase matches for both.
        hit_upper = len(tgt_up_set.intersection(set(upper)))
        hit_mouse_case = len(tgt_set.intersection(set(mouse_style)))
        hit_mouse_upper = len(tgt_up_set.intersection(set(mouse_style.str.upper())))

        # Prefer mouse-style when it improves exact-case overlap; otherwise fall back to upper.
        prefer_mouse = hit_mouse_case > hit_upper
        # Tie-breaker: if target genes大多不是全大写，则倾向 mouse-style
        frac_target_mixed = float((tgt != tgt_up).mean()) if len(tgt) else 0.0
        if not prefer_mouse and hit_mouse_case == hit_upper and frac_target_mixed >= 0.2:
            prefer_mouse = True

        chosen = mouse_style if prefer_mouse else upper
        try:
            _logging.getLogger('cellscope').info(
                "Gold gene case harmonization: upper=%d, mouse_case=%d, mouse_upper=%d, target_mixed=%.2f, chosen=%s",
                hit_upper, hit_mouse_case, hit_mouse_upper, frac_target_mixed,
                'mouse-style' if prefer_mouse else 'upper'
            )
        except Exception:
            pass

    gd['Common_Gene'] = chosen

    # Deduplicate after recasing; aggregate by max to keep conservative probabilities.
    if gd['Common_Gene'].duplicated().any():
        num_cols = [c for c in gd.columns if c != 'Common_Gene']
        agg = gd.groupby('Common_Gene', sort=False)[num_cols].max()
        gd = agg.reset_index()

    return gd


def _coerce_bool_series(values: pd.Series, default: bool = True) -> pd.Series:
    """Parse bool-like values, including categorical "True"/"False" strings."""
    raw = pd.Series(values, index=values.index, dtype="object")
    parsed = raw.map(
        {
            True: True,
            False: False,
            1: True,
            0: False,
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "True": True,
            "False": False,
        }
    )
    parsed.loc[parsed.isna()] = bool(default)
    return parsed.astype(bool)


def _prepare_apex_gold_reference(
    apex_raw: pd.DataFrame,
    *,
    omm_specificity_filter: bool = True,
    omm_min_log2fc: float = 0.75,
    omm_min_delta: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the bundled APEX reference and audit OMM specificity.

    APEX-seq enrichment is a soft localization reference, not a census of
    molecules on an organelle.  In particular, the published OMM and ERM
    programs share many secretory-pathway RNAs.  An RNA is therefore allowed
    into the *OMM-specific* reference only when its raw OMM enrichment clears
    both a basal threshold and the stronger of ERM/Cytosol by ``omm_min_delta``.

    The gate is evaluated on the original log2 fold changes.  Row-wise
    normalization is performed afterwards, and failed rows have only their OMM
    value set to zero.  Their ER, Cytosol, and other localization evidence is
    deliberately retained.
    """
    required = {
        "Common_Gene",
        "OMM_log2FC",
        "ERM_log2FC",
        "Cytosol_log2FC",
    }
    missing = sorted(required - set(apex_raw.columns))
    if missing:
        raise ValueError(
            "Bundled APEX reference is missing raw columns required for the "
            f"OMM specificity audit: {missing}"
        )
    try:
        min_log2fc = float(omm_min_log2fc)
        min_delta = float(omm_min_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "APEX OMM thresholds must be finite numeric values"
        ) from exc
    if not np.isfinite(min_log2fc) or not np.isfinite(min_delta):
        raise ValueError("APEX OMM thresholds must be finite numeric values")
    if min_delta < 0.0:
        raise ValueError("apex_omm_min_delta must be non-negative")

    raw = apex_raw.copy()
    omm = pd.to_numeric(raw["OMM_log2FC"], errors="coerce")
    erm = pd.to_numeric(raw["ERM_log2FC"], errors="coerce")
    cytosol = pd.to_numeric(raw["Cytosol_log2FC"], errors="coerce")
    valid = omm.notna() & erm.notna() & cytosol.notna()
    competitor = pd.concat(
        [erm.rename("ERM"), cytosol.rename("Cytosol")],
        axis=1,
    ).max(axis=1)
    delta = omm - competitor
    basal_pass = valid & omm.ge(min_log2fc)
    specific_pass = basal_pass & delta.ge(min_delta)
    shared = basal_pass & ~specific_pass

    reason = pd.Series("invalid_numeric", index=raw.index, dtype="object")
    reason.loc[valid & ~basal_pass] = "below_basal"
    reason.loc[shared] = "shared_with_erm_or_cytosol"
    reason.loc[specific_pass] = "specific"
    audit = pd.DataFrame(
        {
            "Common_Gene": raw["Common_Gene"],
            "Source": raw.get(
                "Source",
                pd.Series("", index=raw.index, dtype="object"),
            ),
            "OMM_log2FC": omm,
            "ERM_log2FC": erm,
            "Cytosol_log2FC": cytosol,
            "omm_competitor_log2fc": competitor,
            "omm_specificity_delta": delta,
            "omm_basal_pass": basal_pass.astype(bool),
            "omm_specific_pass": specific_pass.astype(bool),
            "omm_shared": shared.astype(bool),
            "omm_filter_reason": reason,
        }
    ).reset_index(drop=True)

    cols = ["Common_Gene"] + [
        column
        for column in raw.columns
        if str(column).endswith("_log2FC")
    ]
    gold = raw[cols].rename(
        columns={
            column: column.replace("_log2FC", "")
            for column in cols
            if column != "Common_Gene"
        }
    )
    complete = gold.notna().all(axis=1)
    gold = gold.loc[complete].copy()
    values_columns = [
        column for column in gold.columns if column != "Common_Gene"
    ]
    values = (
        gold[values_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    values = values.subtract(values.min(axis=1), axis=0)
    span = (values.max(axis=1) - values.min(axis=1)).replace(0, 1.0)
    values = values.div(span, axis=0)
    if bool(omm_specificity_filter) and "OMM" in values.columns:
        values.loc[~specific_pass.reindex(values.index).fillna(False), "OMM"] = 0.0
    gold = pd.concat(
        [
            gold[["Common_Gene"]].reset_index(drop=True),
            values.reset_index(drop=True),
        ],
        axis=1,
    )
    return gold, audit


def _load_default_gold_tables(
    *,
    apex_omm_specificity_filter: bool = True,
    apex_omm_min_log2fc: float = 0.75,
    apex_omm_min_delta: float = 0.50,
    return_omm_audit: bool = False,
):
    apex = bundled_resource_path("2019_CELL_APEXSeq.tsv")
    rna = bundled_resource_path("RNALocateV3.txt")
    if not os.path.exists(apex) or not os.path.exists(rna):
        raise FileNotFoundError(
            "Bundled APEX/RNALocate reference files are missing from the "
            "CellScoper installation"
        )
    apex_raw = pd.read_table(apex)
    rna_locate = pd.read_table(rna)
    gold_apex, omm_audit = _prepare_apex_gold_reference(
        apex_raw,
        omm_specificity_filter=apex_omm_specificity_filter,
        omm_min_log2fc=apex_omm_min_log2fc,
        omm_min_delta=apex_omm_min_delta,
    )
    if return_omm_audit:
        return gold_apex, rna_locate, omm_audit
    return gold_apex, rna_locate


def _load_reference_scores(
    gold_data: Optional[Union[str, pd.DataFrame]],
    annotation_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Gene x compartment reference scores, indexed by upper-case symbol.

    Uses the same sources and hold-out option as RAP-X: a user table when
    given, otherwise bundled APEX-seq merged with RNALocate.
    """
    if gold_data is not None:
        table = (
            gold_data.copy()
            if isinstance(gold_data, pd.DataFrame)
            else (
                pd.read_parquet(gold_data)
                if str(gold_data).lower().endswith((".parquet", ".pq"))
                else pd.read_csv(gold_data)
            )
        )
        if "Common_Gene" not in table.columns:
            table = table.reset_index()
        if "Common_Gene" not in table.columns:
            raise ValueError("gold_data must contain a 'Common_Gene' column")
        reference = merge_gold(table, table.iloc[0:0], fuse="max")
        # Same per-gene 0-1 scaling RAP-X applies to a user table.
        reference = reference.sub(reference.min(axis=1), axis=0)
        reference = reference.div(reference.max(axis=1).replace(0, 1.0), axis=0)
    else:
        gold_apex, rna_locate = _load_default_gold_tables(
            apex_omm_specificity_filter=bool(
                annotation_cfg.get("apex_omm_specificity_filter", True)
            ),
            apex_omm_min_log2fc=float(annotation_cfg.get("apex_omm_min_log2fc", 0.75)),
            apex_omm_min_delta=float(annotation_cfg.get("apex_omm_min_delta", 0.50)),
        )
        species = annotation_cfg.get(
            "rnalocate_species", ("Homo sapiens", "Mus musculus")
        )
        if isinstance(species, str):
            species = tuple(v.strip() for v in species.split(",") if v.strip())
        from_rnalocate = _configure_rnalocate_augmentation(
            project_l1_to_seven(
                build_gold_from_rnalocate_l1(
                    rna_locate,
                    species=tuple(species) if species is not None else None,
                    agg_mode="prob_or",
                    score_floor=0.6,
                    use_pubmed_weight=True,
                ),
                mode="strict",
                mito_non_omm=None,
            ),
            augment_er=bool(annotation_cfg.get("rnalocate_augment_er", False)),
        )
        reference = merge_gold(gold_apex, from_rnalocate, fuse="max")
    reference.index = reference.index.astype(str).str.upper()
    reference = reference.loc[~reference.index.duplicated()]
    exclude_file = annotation_cfg.get("reference_exclude_genes_file")
    if exclude_file:
        with open(str(exclude_file), "r", encoding="utf-8") as handle:
            excluded = {line.strip().upper() for line in handle if line.strip()}
        kept = ~reference.index.isin(excluded)
        _logging.getLogger("cellscope").info(
            "Reference hold-out: excluded %d of %d reference genes (%d listed)",
            int((~kept).sum()),
            int(kept.size),
            len(excluded),
        )
        reference = reference.loc[kept]
    return reference


def _annotate_with_evidence(
    adata1: ad.AnnData,
    final_df: pd.DataFrame,
    gold_data: Optional[Union[str, pd.DataFrame]],
    out_dir: Optional[str],
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Run the evidence annotation and write its columns and report."""
    import json

    from .evidence_annotation import run_evidence_annotation

    log = _logging.getLogger("cellscope")
    annotation_cfg = resolve_annotation_configuration(load_params_yaml() or {})
    reference = _load_reference_scores(gold_data, annotation_cfg)
    table, report = run_evidence_annotation(final_df, reference, annotation_cfg)
    table = table.set_index("metaspot")

    # MetaDomain summary: the most common primary label of its MetaSpots,
    # separately on each side of the nuclear boundary.
    spot_domain = (
        final_df.drop_duplicates("metaspot")
        .set_index("metaspot")["meta-domain"]
        .astype(str)
        .reindex(table.index)
    )
    domain_key = spot_domain + "|" + table["subcellular_broad_compartment"].astype(str)
    table["subcellular_annotation_domain"] = (
        table["subcellular_annotation_validated"]
        .groupby(domain_key.to_numpy())
        .transform(lambda labels: labels.value_counts().index[0])
    )

    rows = table.index.get_indexer(final_df["metaspot"])
    final_df = _replace_frame_columns(
        final_df, {name: table[name].to_numpy()[rows] for name in table.columns}
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
    return adata1, final_df


def _configure_rnalocate_augmentation(
    gold_from_rna: pd.DataFrame,
    augment_er: bool = False,
) -> pd.DataFrame:
    """Keep broad RNALocate ER associations from overriding APEX specificity."""
    out = gold_from_rna.copy()
    if not augment_er:
        for column in ("ER", "ERM", "ER_Lumen"):
            if column in out.columns:
                out[column] = 0.0
    return out


