import numpy as np
import pandas as pd
import pytest

from cellscoper.scalable_execution import resolve_execution_plan
from cellscoper.checkpoints import _load_module3_membership_checkpoint, _save_module3_membership_checkpoint
from cellscoper.outputs import _atomic_partitioned_parquet_write


def _config(profile="auto"):
    return {"execution": {"profile": profile, "exact_max_metaspots": 100, "sampled_max_metaspots": 1_000}}


@pytest.mark.parametrize(
    ("nodes", "profile", "partitioned"),
    [(50, "exact", False), (500, "sampled", False), (5_000, "streaming", True)],
)
def test_execution_plan_routes_by_realized_metaspot_count(nodes, profile, partitioned):
    plan = resolve_execution_plan(_config(), n_metaspots=nodes)
    assert plan.effective_profile == profile
    assert plan.partitioned_annotation is partitioned
    assert plan.full_diagnostics is not partitioned


def test_exact_profile_refuses_more_metaspots_than_its_limit():
    with pytest.raises(ValueError):
        resolve_execution_plan(_config("exact"), n_metaspots=500)


def test_partitioned_parquet_keeps_each_cell_in_one_part(tmp_path):
    frame = pd.DataFrame(
        {
            "transcript_id": np.arange(17),
            "cell_id": np.repeat(["a", "b", "c", "d"], [5, 4, 3, 5]),
        }
    )
    destination = tmp_path / "final_data_dataset"
    manifest = _atomic_partitioned_parquet_write(
        frame,
        str(destination),
        target_rows_per_part=5,
    )
    assert manifest["rows"] == len(frame)
    assert (destination / "_SUCCESS").is_file()
    memberships = {}
    for part in destination.glob("part-*.parquet"):
        for cell in pd.read_parquet(part)["cell_id"].unique():
            memberships.setdefault(cell, set()).add(part.name)
    assert all(len(parts) == 1 for parts in memberships.values())


def test_partitioned_parquet_cell_hash_is_independent_of_row_order(tmp_path):
    frame = pd.DataFrame(
        {
            "transcript_id": np.arange(24),
            "cell_id": np.repeat(["a", "b", "c", "d"], 6),
        }
    )
    first = _atomic_partitioned_parquet_write(
        frame,
        str(tmp_path / "first"),
        target_rows_per_part=6,
    )
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = _atomic_partitioned_parquet_write(
        shuffled,
        str(tmp_path / "second"),
        target_rows_per_part=6,
    )

    def cell_parts(destination):
        result = {}
        for part in destination.glob("part-*.parquet"):
            for cell in pd.read_parquet(part)["cell_id"].unique():
                result[cell] = part.name
        return result

    assert first["n_parts"] == second["n_parts"]
    assert cell_parts(tmp_path / "first") == cell_parts(tmp_path / "second")


def test_compact_module3_membership_restores_only_when_rows_align(tmp_path):
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b", "b"],
            "feature_name": ["G1", "G2", "G1", "G2"],
            "metaspot_in_cell": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "metaspot": ["a:0", "a:1", "b:0", "b:1"],
        }
    )
    fingerprints = {"m1": "one", "m2": "two", "m3": "three"}
    _save_module3_membership_checkpoint(
        frame,
        str(tmp_path),
        module_fingerprints=fingerprints,
    )
    upstream = frame.drop(columns=["metaspot_in_cell", "metaspot"])
    restored = _load_module3_membership_checkpoint(
        upstream,
        str(tmp_path),
        module_fingerprints=fingerprints,
    )
    assert restored is not None
    assert restored["metaspot"].astype(str).tolist() == frame["metaspot"].tolist()
    misaligned = upstream.iloc[[0, 2, 1, 3]].reset_index(drop=True)
    assert (
        _load_module3_membership_checkpoint(
            misaligned,
            str(tmp_path),
            module_fingerprints=fingerprints,
        )
        is None
    )


def test_three_dimensional_metaspots_split_layers_that_overlap_in_xy():
    from cellscoper.metaspots import _enforce_spatially_connected_labels

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 3, size=(200, 2)).astype(np.float32)
    lower, upper = np.c_[xy, np.zeros(200)], np.c_[xy, np.full(200, 4.0)]
    points = np.vstack([lower, upper]).astype(np.float32)
    features = rng.normal(size=(400, 3)).astype(np.float32)
    labels = _enforce_spatially_connected_labels(
        points, np.zeros(400, dtype=np.int32), feature_matrix=features, target_n_clusters=2
    )
    assert set(labels[:200]).isdisjoint(labels[200:])
    flat = _enforce_spatially_connected_labels(
        points[:, :2], np.zeros(400, dtype=np.int32), feature_matrix=features, target_n_clusters=2
    )
    assert flat.shape == (400,)
