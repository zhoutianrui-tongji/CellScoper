"""Process runtime settings shared by the pipeline modules: BLAS/OpenMP/Numba thread limits (set before numpy is
imported), worker counts, joblib backend, progress printing, and small table helpers."""

from typing import Any, Dict, Mapping, Optional, Tuple
import os
import tempfile
import warnings as _warnings

# Numba and Matplotlib otherwise try to cache beside read-only site-packages or
# under a non-writable home in containers/HPC jobs. Keep caches process-local
# and writable without changing numerical behavior.
_runtime_cache_root = os.path.join(tempfile.gettempdir(), "cellscoper-cache")
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
    _coerce_pos_int(os.environ.get("CELLSCOPER_N_JOBS"))
    or _coerce_pos_int(_runtime_cfg_threads.get("n_jobs"))
    or 1
)
_threaded_outer_parallelism = (
    str(
        os.environ.get("CELLSCOPER_JOBLIB_BACKEND")
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

from sklearn.neighbors import KDTree
import anndata as ad
import numpy as np
import pandas as pd


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

def _preferred_n_jobs() -> int:
    """Return preferred parallel worker count.

    Env vars (checked in order):
      CELLSCOPER_N_JOBS         -> explicit override

    params.yaml (runtime.n_jobs) is honored when env vars are absent.

    Fallback heuristic: about half of the available CPUs, at most 32.
    """
    for k in ("CELLSCOPER_N_JOBS",):
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
    # Users can override via CELLSCOPER_JOBLIB_BACKEND env or runtime.joblib_backend.
    env_backend = os.environ.get("CELLSCOPER_JOBLIB_BACKEND")
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

    Env: CELLSCOPER_CLUSTER_CHUNK (int)
    Config: module3.parallel_chunk_cells (with module2's legacy setting as a
    fallback when called outside the pipeline).
    Smaller chunks reduce peak memory (fewer simultaneous process payloads)
    at the cost of higher scheduler overhead.
    """
    try:
        env_v = os.environ.get("CELLSCOPER_CLUSTER_CHUNK", "").strip()
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
        v = int(os.environ.get("CELLSCOPER_CLUSTER_CHUNK", "512"))
        return max(128, min(4096, v))
    except Exception:
        return 256

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
