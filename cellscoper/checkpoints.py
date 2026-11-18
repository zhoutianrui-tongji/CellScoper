"""Intermediate checkpoints, run state and atomic writes: what a resumed run may reuse, and how files are written so
an interrupted run never leaves a half-written output."""

from typing import Any, Mapping, Optional, Tuple
import ctypes
import gc
import logging as _logging
import os
import tempfile
import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import load_params_yaml
from .io import load_checkpoint
from .keys import COMPACT_STRING_COLS, LENS_REP, METASPOT_COL, METASPOT_LOCAL_COL
from .runtime import _replace_frame_columns
from .metaspots import _validate_adata1_point_df_alignment
from .provenance import atomic_write_json


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
            return pd.read_parquet(pq_path)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
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
            _logging.getLogger("cellscoper").warning(
                "Rejecting compact Module 3 membership: upstream module "
                "fingerprints differ"
            )
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(parquet_path)
        row_count = int(parquet.metadata.num_rows)
        if row_count != len(point_df) or int(manifest.get("rows", -1)) != row_count:
            _logging.getLogger("cellscoper").warning(
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
        logger = _logging.getLogger("cellscoper")
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
            _logging.getLogger("cellscoper").warning(
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
        _logging.getLogger("cellscoper").warning(
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
        _logging.getLogger("cellscoper").warning(
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
        _logging.getLogger("cellscoper").warning(
            "Rejecting late point-table resume checkpoint: %s",
            reason,
        )
        return None
    return point_df

def _load_adata_checkpoint(path: str) -> Optional[ad.AnnData]:
    if not os.path.exists(path):
        return None
    try:
        return ad.read_h5ad(path)
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
    if adata2 is None or LENS_REP not in adata1.obsm:
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
        _logging.getLogger("cellscoper").info(
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

def _checkpoint_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f".{name}.parquet")

# Global I/O switches for intermediates, set by run_pipeline and/or params.yaml
# Process-wide I/O switches for the helpers below, set by run_pipeline for the duration of a run (None = read
# params.yaml).
_WRITE_INTERMEDIATE: Optional[bool] = None
_LITE_INTERMEDIATE: Optional[bool] = None
_WRITE_FIGURES: Optional[bool] = None


def io_switches() -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """Current (write_intermediate, lite_intermediate, write_figures) overrides."""
    return _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES


def set_io_switches(write_intermediate: Optional[bool], lite_intermediate: Optional[bool],
                    write_figures: Optional[bool]) -> None:
    """Override the I/O switches (None = follow params.yaml)."""
    global _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES
    _WRITE_INTERMEDIATE, _LITE_INTERMEDIATE, _WRITE_FIGURES = write_intermediate, lite_intermediate, write_figures

def _io_should_write_intermediate() -> bool:
    """Decide whether to write intermediate artifacts.

    Priority:
      1) explicit global override set by run_pipeline
      2) io.write_intermediate (if present in params.yaml)
      3) default True
    """
    global _WRITE_INTERMEDIATE
    if _WRITE_INTERMEDIATE is not None:
        return bool(_WRITE_INTERMEDIATE)
    try:
        _cfg = load_params_yaml()
        io_cfg = _cfg.get('io', {}) if _cfg else {}
        if 'write_intermediate' in io_cfg:
            return bool(io_cfg.get('write_intermediate'))
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

def _release_freed_memory() -> None:
    """把已释放的内存还给系统（glibc 默认留在进程里）；只影响内存占用。"""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def compact_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """转录本表的重复文字列就地改为 categorical（类别按字典序），取值不变。"""
    for col in COMPACT_STRING_COLS:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    _release_freed_memory()
    return df


def _plain_strings(df: pd.DataFrame) -> pd.DataFrame:
    """把 compact_string_columns 改过的列还原成普通文字（共用类别里的字符串对象，只占指针）。"""
    columns = {c: df[c].astype(object) for c in COMPACT_STRING_COLS if c in df.columns and isinstance(df[c].dtype, pd.CategoricalDtype)}
    return df.assign(**columns) if columns else df


def _write_parquet_batches(df: pd.DataFrame, path: str, batch_rows: int = 1024 * 1024) -> None:
    """按 row group 分批写 Parquet：每批才还原文字列，文件的列类型与行组大小和 DataFrame.to_parquet 相同。"""
    if not any(c in df.columns and isinstance(df[c].dtype, pd.CategoricalDtype) for c in COMPACT_STRING_COLS):
        df.to_parquet(path, index=False)
        return
    writer = None
    try:
        for start in range(0, len(df), batch_rows):
            table = pa.Table.from_pandas(_plain_strings(df.iloc[start:start + batch_rows]), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def write_annotated_transcripts(module5_parquet: str, spot_table: pd.DataFrame, path: str, category_orders: Mapping[str, Any]) -> None:
    """低内存写 final_data.parquet：按行组读 module5 检查点，按 MetaSpot 补上注释列，再分批写出。
    spot_table：注释结果，每个 MetaSpot 一行，index 是 MetaSpot。category_orders：与内存表一致的类别顺序。
    行顺序、列顺序、列类型和行组大小与 _atomic_dataframe_write 写完整内存表相同。"""
    source = pq.ParquetFile(module5_parquet)
    temporary = f"{path}.tmp.{os.getpid()}"
    values = {name: spot_table[name].to_numpy() for name in spot_table.columns}
    writer = None
    try:
        for group in range(source.metadata.num_row_groups):
            batch = source.read_row_group(group).to_pandas()
            for col, categories in category_orders.items():
                if col in batch.columns:
                    batch[col] = pd.Categorical(batch[col], categories=categories)
            rows = spot_table.index.get_indexer(batch[METASPOT_COL].astype(object).to_numpy())
            if (rows < 0).any():
                raise RuntimeError("module5_final_data has MetaSpots missing from the annotation table")
            batch = _replace_frame_columns(batch, {name: column[rows] for name, column in values.items()})
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


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
            _write_parquet_batches(df, temporary)
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
        _logging.getLogger("cellscoper").warning(
            "Could not read I/O settings for table %s: %s; using defaults",
            name,
            exc,
        )
    if force_parquet:
        write_pq = True
    if not write_csv and not write_pq:
        _logging.getLogger("cellscoper").debug(
            "Intermediate table disabled by I/O config: %s",
            name,
        )
        return
    _logging.getLogger("cellscoper").info(
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
            _logging.getLogger("cellscoper").warning(
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
            _logging.getLogger("cellscoper").warning(
                "Failed to save intermediate Parquet %s: %s",
                pq_path,
                exc,
            )
    if written:
        _logging.getLogger("cellscoper").info(
            "Saved %s table | name=%s | rows=%d | columns=%d | files=%s",
            "diagnostic" if diagnostic else "intermediate",
            name,
            len(df),
            int(df.shape[1]),
            ", ".join(written),
        )

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
        _logging.getLogger("cellscoper").warning(
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
        _logging.getLogger("cellscoper").warning(
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

def _save_adata(adata: ad.AnnData, out_dir: str, name: str, compression: str = "lzf"):
    if not _io_should_write_intermediate():
        return
    tmp_path = None
    original_obs = adata.obs
    try:
        p = os.path.join(_intermediate_dir(out_dir), f"{name}.h5ad")
        tmp_path = f"{p}.tmp-{os.getpid()}"
        _logging.getLogger("cellscoper").info(
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
        _logging.getLogger("cellscoper").info(
            "Saved intermediate AnnData: %s",
            p,
        )
    except Exception as exc:
        _logging.getLogger("cellscoper").warning(
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

def _save_fig(fig, out_dir: str, name: str, *, dpi: int = 150, folder: Optional[str] = None):
    """原子写 PNG；folder 为 None 时写到中间结果目录，否则写到 out_dir/folder。"""
    if not _io_should_write_figures():
        return
    temporary = None
    try:
        if folder is None:
            base = _intermediate_dir(out_dir)
        else:
            base = os.path.join(out_dir, folder)
            os.makedirs(base, exist_ok=True)
        p = os.path.join(base, f"{name}.png")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp.png",
            dir=os.path.dirname(p),
        )
        os.close(fd)
        fig.savefig(temporary, bbox_inches='tight', dpi=int(dpi))
        os.replace(temporary, p)
        temporary = None
        _logging.getLogger("cellscoper").info(
            "Saved diagnostic figure | name=%s | path=%s",
            name,
            p,
        )
    except Exception as exc:
        _logging.getLogger("cellscoper").warning(
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
