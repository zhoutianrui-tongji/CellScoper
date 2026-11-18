import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from cellscoper.config import (
    ANNOTATION_SCHEMA_VERSION,
    fingerprint_module,
    fingerprint_stage,
    get_default_params_path,
    load_params_yaml,
    stage_config_subset,
)
from cellscoper.io import read_boundaries, read_spatial_transcripts


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_input_readers_accept_csv_and_parquet_with_optional_nucleus_flags(
    tmp_path,
    suffix,
):
    spatial_path = tmp_path / f"spatial{suffix}"
    boundary_path = tmp_path / f"boundary{suffix}"
    spatial = pd.DataFrame(
        {
            "cell_id": ["c1", "c1"],
            "x_location": [1.0, 2.0],
            "y_location": [3.0, 4.0],
            "feature_name": ["G1", "G2"],
        }
    )
    boundary = pd.DataFrame(
        {
            "cell_id": ["c1"] * 3,
            "vertex_x": [0.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0],
        }
    )
    if suffix == ".csv":
        spatial.to_csv(spatial_path, index=False)
        boundary.to_csv(boundary_path, index=False)
    else:
        spatial.to_parquet(spatial_path, index=False)
        boundary.to_parquet(boundary_path, index=False)

    loaded_spatial = read_spatial_transcripts(str(spatial_path))
    loaded_boundary = read_boundaries(str(boundary_path))

    assert "overlaps_nucleus" not in loaded_spatial.columns
    assert loaded_spatial["x_location"].dtype == np.dtype("float32")
    assert loaded_boundary["vertex_x"].dtype == np.dtype("float32")
    assert loaded_spatial["cell_id"].astype(str).tolist() == ["c1", "c1"]


def test_csv_input_readers_preserve_cell_ids_without_overflow_or_suffix_loss(
    tmp_path,
):
    spatial_path = tmp_path / "spatial.csv"
    boundary_path = tmp_path / "boundary.csv"
    cell_ids = ["4294967296", "A.0", "sample.000", "001"]
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_location": np.arange(4, dtype=float),
            "y_location": np.arange(4, dtype=float),
            "feature_name": ["G1"] * 4,
        }
    ).to_csv(spatial_path, index=False)
    pd.DataFrame(
        {
            "cell_id": cell_ids,
            "vertex_x": np.arange(4, dtype=float),
            "vertex_y": np.arange(4, dtype=float),
        }
    ).to_csv(boundary_path, index=False)

    spatial = read_spatial_transcripts(str(spatial_path))
    boundaries = read_boundaries(str(boundary_path))

    assert spatial["cell_id"].tolist() == cell_ids
    assert boundaries["cell_id"].tolist() == cell_ids


def test_pipeline_import_uses_single_blas_thread_with_threading_backend(tmp_path):
    config = tmp_path / "params.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "blas_threads": 12,
                    "numba_threads": 12,
                    "scanpy_n_jobs": 12,
                    "n_jobs": 12,
                    "joblib_backend": "threading",
                }
            }
        ),
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "CELLSCOPER_CONFIG": str(config),
            "OPENBLAS_NUM_THREADS": "128",
            "OMP_NUM_THREADS": "128",
            "MKL_NUM_THREADS": "128",
            "NUMEXPR_MAX_THREADS": "128",
            "NUMEXPR_NUM_THREADS": "128",
            "NUMBA_NUM_THREADS": "128",
            "SCANPY_N_JOBS": "128",
            "PYTHONPATH": str(package_root),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; import cellscoper.cli; "
                "assert 'cellscoper.pipeline' not in sys.modules; "
                "import cellscoper.pipeline; import numba; "
                "print('|'.join(os.environ[name] for name in "
                "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', "
                "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', "
                "'NUMBA_NUM_THREADS')) + '|' + str(numba.get_num_threads()))"
            ),
        ],
        cwd=str(package_root),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )

    assert result.stdout.strip().splitlines()[-1] == "1|1|1|1|5|5"


def test_default_annotation_config_resolves_production_defaults():
    config = load_params_yaml(path=get_default_params_path())
    public = config["annotation"]

    assert config["module3"]["parallel_n_jobs"] == 12
    assert config["module5"]["metadomain_q"] == 0.2
    assert "fixed_k" not in config["module5"] and "lens_weight" not in config["module5"]
    assert config["io"]["cell_structure_features"] is True
    assert "module7" not in config and "module8" not in config

    assert public["method"] == "evidence"
    assert config["module1"]["remove_foreign_transcripts"] is True
    assert set(public) - {"enable", "method"} == {
        key for key in public if key.startswith("evidence_")
    }


def test_stage_fingerprint_changes_with_input_provenance():
    config = {"module3": {"target_sites_per_metaspot": 15}}
    first = fingerprint_stage(
        config,
        "build_anndata",
        context={"inputs": {"spatial": {"sha256": "a"}}},
    )
    second = fingerprint_stage(
        config,
        "build_anndata",
        context={"inputs": {"spatial": {"sha256": "b"}}},
    )
    assert first != second


def test_output_settings_do_not_invalidate_input_loading():
    config = {
        "io": {
            "final_format": "csv",
            "h5ad_compression": "lzf",
        }
    }
    changed = copy.deepcopy(config)
    changed["io"]["final_format"] = "parquet"
    changed["io"]["h5ad_compression"] = "gzip"

    assert fingerprint_stage(config, "load_inputs") == fingerprint_stage(
        changed,
        "load_inputs",
    )
    assert fingerprint_stage(config, "write_outputs") != fingerprint_stage(
        changed,
        "write_outputs",
    )


def test_final_column_pruning_invalidates_build_stage():
    config = {"io": {"drop_embeddings": False}}
    changed = {"io": {"drop_embeddings": True}}

    assert fingerprint_stage(config, "build_anndata") != fingerprint_stage(
        changed,
        "build_anndata",
    )
    assert fingerprint_stage(config, "write_outputs") != fingerprint_stage(
        changed,
        "write_outputs",
    )


def test_annotation_schema_version_is_internal_to_fingerprint():
    subset = stage_config_subset({"annotation": {}}, "annotate")

    assert (
        subset["__annotation_schema_version__"]
        == ANNOTATION_SCHEMA_VERSION
    )


def test_metadomain_random_seed_invalidates_build_and_module5():
    current = load_params_yaml(path=get_default_params_path())
    changed = copy.deepcopy(current)
    changed["module5"]["random_seed"] += 1

    assert fingerprint_stage(current, "build_anndata") != fingerprint_stage(
        changed,
        "build_anndata",
    )
    assert fingerprint_module(current, "m5") != fingerprint_module(
        changed,
        "m5",
    )
