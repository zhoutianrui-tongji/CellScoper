import os
import tempfile
import unittest
import warnings
from unittest import mock
from contextlib import contextmanager

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import cellscoper.metaspots as metaspots_module

from cellscoper.checkpoints import _sanitize_obs_for_h5ad, _resume_build_outputs, _save_adata
from cellscoper.composition import _LocalCompositionModel, build_composition_embeddings
from cellscoper.geometry import _stretch_signed_geometry, build_point_geometry_v2
from cellscoper.metadomains import build_adata2_from_adata1_and_pointdf
from cellscoper.metaspots import _allocate_partition_target_counts, _enforce_spatially_connected_labels, _metaspot_segmentation_audit, _select_top_variance_features, _target_metaspot_count, _validate_adata1_point_df_alignment, build_adata1, per_cell_clustering
from cellscoper.reference import _configure_rnalocate_augmentation, _prepare_apex_gold_reference, _load_default_gold_tables
from cellscoper.runtime import _coerce_bool_series
from cellscoper.config import module_config_subset, stage_config_subset


@contextmanager
def _local_config():
    content = """
module2:
  top_g: 4
  bow_knn_k: 5
  max_rows_per_cell_for_svd: 40
  max_svd_rows: 160
  svd_n_components: 3
  svd_sample_cells: 4
  use_tfidf: true
  tf_norm: true
runtime:
  n_jobs: 1
  joblib_backend: threading
"""
    old = os.environ.get("CELLSCOPER_CONFIG")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(content)
        path = handle.name
    os.environ["CELLSCOPER_CONFIG"] = path
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CELLSCOPER_CONFIG", None)
        else:
            os.environ["CELLSCOPER_CONFIG"] = old
        os.unlink(path)


def _synthetic_transcripts():
    rows = []
    for cell in range(4):
        for i in range(40):
            left = i < 20
            genes = ("LEFT", "RIGHT") if cell % 2 == 0 else ("A", "B")
            rows.append(
                {
                    "cell_id": str(cell),
                    "x_location": float(i),
                    "y_location": float(cell),
                    "feature_name": genes[0] if left else genes[1],
                    "r_signed": 0.0,
                    "d_cell_norm": 0.5,
                    "d_nuc_norm": 0.5,
                }
            )
    return pd.DataFrame(rows)


class LocalCompositionTests(unittest.TestCase):
    def test_module6_preserves_raw_counts_separately_from_tf_mean(self):
        obs = pd.DataFrame(
            {
                "meta-domain": pd.Categorical(["A", "A", "B"]),
                "cell_id": ["c1", "c2", "c1"],
                "r_mean": [-0.5, -0.4, 0.6],
                "r_std": [0.1, 0.1, 0.2],
                "r_p50": [-0.5, -0.4, 0.6],
                "d_cell_norm_mean": [0.4, 0.4, 0.3],
                "d_nuc_norm_mean": [0.2, 0.2, 0.7],
                "elongation": [1.0, 1.1, 1.2],
                "spread": [0.3, 0.4, 0.5],
                "log_n": [1.0, 1.2, 1.1],
            },
            index=["m1", "m2", "m3"],
        )
        adata1 = ad.AnnData(
            X=sp.csr_matrix([[2, 0], [0, 3], [1, 1]], dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=["G1", "G2"]),
        )

        result = build_adata2_from_adata1_and_pointdf(
            adata1,
            pd.DataFrame(),
        )

        np.testing.assert_allclose(
            result.layers["counts"].toarray(),
            np.array([[2, 3], [1, 1]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(result.X.sum(axis=1)).ravel(),
            np.ones(2),
        )
        counts = result.layers["counts"]
        self.assertTrue(np.allclose(counts.data, np.round(counts.data)) and (counts.data >= 0).all())
        self.assertIn(
            "layers_counts",
            result.uns["matrix_semantics"],
        )

    def test_module6_accepts_lightweight_geometry_metadata(self):
        obs = pd.DataFrame(
            {
                "meta-domain": pd.Categorical(["A", "A", "B"]),
                "cell_id": ["c1", "c1", "c2"],
                "r_mean": [-0.5, -0.4, 0.6],
                "r_p50": [-0.5, -0.4, 0.6],
                "d_cell_norm_mean": [0.4, 0.4, 0.3],
                "d_nuc_norm_mean": [0.2, 0.2, 0.7],
            },
            index=["m1", "m2", "m3"],
        )
        adata1 = ad.AnnData(
            X=sp.csr_matrix([[2, 0], [0, 3], [1, 1]], dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=["G1", "G2"]),
        )

        result = build_adata2_from_adata1_and_pointdf(
            adata1,
            pd.DataFrame(),
        )

        assert list(result.obs.columns) == [
            "n_metaspots",
            "r_mean",
            "r_p50",
            "d_cell_norm_mean",
            "d_nuc_norm_mean",
            "rep_cell_id",
        ]
        np.testing.assert_allclose(
            result.obs["r_mean"].to_numpy(),
            np.array([-0.45, 0.6]),
        )


    def test_point_geometry_prefers_input_nucleus_flags_for_unordered_boundaries(self):
        transcripts = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "x_location": [0.25, 0.75, 0.25, 0.75],
                "y_location": [0.25, 0.25, 0.75, 0.75],
                "feature_name": ["A", "A", "B", "B"],
                "overlaps_nucleus": [1, 0, 1, 0],
            }
        )
        # Deliberately unordered raster-like boundary points.
        nucleus = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [0.0, 1.0, 0.0, 1.0],
                "vertex_y": [0.0, 1.0, 1.0, 0.0],
            }
        )
        cell = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [-1.0, 2.0, 2.0, -1.0],
                "vertex_y": [-1.0, -1.0, 2.0, 2.0],
            }
        )

        result = build_point_geometry_v2(
            transcripts,
            nucleus,
            cell,
            sampled_cells=["cell"],
            include_directional=False,
        )

        self.assertEqual(result["in_nucleus"].tolist(), [True, False, True, False])
        self.assertTrue((result.loc[result["in_nucleus"], "r_signed"] <= 0).all())
        self.assertTrue((result.loc[~result["in_nucleus"], "r_signed"] >= 0).all())

    def test_point_geometry_can_recompute_nucleus_flags_from_polygon(self):
        transcripts = pd.DataFrame(
            {
                "cell_id": ["cell", "cell"],
                "x_location": [0.5, 1.5],
                "y_location": [0.5, 0.5],
                "feature_name": ["A", "B"],
                # Deliberately wrong; benchmark mode must ignore this column.
                "overlaps_nucleus": [0, 1],
            }
        )
        nucleus = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [0.0, 1.0, 1.0, 0.0],
                "vertex_y": [0.0, 0.0, 1.0, 1.0],
            }
        )
        cell = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [-1.0, 2.0, 2.0, -1.0],
                "vertex_y": [-1.0, -1.0, 2.0, 2.0],
            }
        )

        result = build_point_geometry_v2(
            transcripts,
            nucleus,
            cell,
            sampled_cells=["cell"],
            include_directional=False,
            use_input_nucleus_flags=False,
        )

        self.assertEqual(result["in_nucleus"].tolist(), [True, False])

    def test_point_geometry_edge_distance_uses_polygon_edges(self):
        transcripts = pd.DataFrame(
            {
                "cell_id": ["cell", "cell"],
                "x_location": [2.0, 0.5],
                "y_location": [0.25, 2.0],
                "feature_name": ["A", "B"],
            }
        )
        nucleus = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [1.0, 3.0, 3.0, 1.0],
                "vertex_y": [1.0, 1.0, 3.0, 3.0],
            }
        )
        cell = pd.DataFrame(
            {
                "cell_id": ["cell"] * 4,
                "vertex_x": [0.0, 4.0, 4.0, 0.0],
                "vertex_y": [0.0, 0.0, 4.0, 4.0],
            }
        )
        kwargs = dict(
            sampled_cells=["cell"],
            include_directional=False,
            use_input_nucleus_flags=False,
            winsor_perc=(0, 100),
        )

        vertex = build_point_geometry_v2(
            transcripts.copy(), nucleus, cell, **kwargs
        )
        edge = build_point_geometry_v2(
            transcripts.copy(), nucleus, cell, boundary_distance="edge", **kwargs
        )

        np.testing.assert_allclose(edge["d_cell"], [0.25, 0.5], atol=1e-6)
        np.testing.assert_allclose(edge["d_nuc"], [0.75, 0.5], atol=1e-6)
        # Mid-edge points are farther from every vertex than from the edge.
        self.assertTrue((vertex["d_cell"] > edge["d_cell"] + 0.5).all())
        self.assertEqual(
            vertex["in_nucleus"].tolist(), edge["in_nucleus"].tolist()
        )
        with self.assertRaises(ValueError):
            build_point_geometry_v2(
                transcripts.copy(), nucleus, cell,
                boundary_distance="nearest", **kwargs
            )

    def test_signed_geometry_stretch_preserves_boundary_sign(self):
        values = np.array([-4.0, -1.0, -0.1, 0.0, 0.1, 1.0, 4.0] * 3)
        stretched = _stretch_signed_geometry(values, cell_size=len(values))

        self.assertTrue((stretched[values < 0] <= 0).all())
        self.assertTrue((stretched[values > 0] >= 0).all())
        self.assertEqual(float(stretched[values == 0][0]), 0.0)


    def test_h5ad_sanitizer_keeps_compact_categoricals(self):
        adata = ad.AnnData(np.zeros((3, 1), dtype=np.float32))
        adata.obs["label"] = pd.Categorical(
            ["Nucleus", "ER", "Nucleus"]
        )
        adata.obs["scalar_text"] = pd.Series(
            ["a", "b", "c"],
            index=adata.obs_names,
            dtype=object,
        )
        adata.obs["sequence"] = pd.Series(
            [["a", "b"], ["c"], None],
            index=adata.obs_names,
            dtype=object,
        )

        _sanitize_obs_for_h5ad(adata)

        self.assertIsInstance(adata.obs["label"].dtype, pd.CategoricalDtype)
        self.assertEqual(adata.obs["scalar_text"].tolist(), ["a", "b", "c"])
        self.assertEqual(adata.obs["sequence"].tolist(), ["a,b", "c", ""])


    def test_rnalocate_er_augmentation_is_disabled_by_default(self):
        gold = pd.DataFrame(
            {
                "Common_Gene": ["A", "B"],
                "ER": [0.8, 0.4],
                "Cytosol": [0.2, 0.7],
            }
        )

        result = _configure_rnalocate_augmentation(gold)

        self.assertEqual(result["ER"].tolist(), [0.0, 0.0])
        self.assertEqual(result["Cytosol"].tolist(), [0.2, 0.7])
        self.assertEqual(gold["ER"].tolist(), [0.8, 0.4])

    def test_rnalocate_er_augmentation_can_be_enabled(self):
        gold = pd.DataFrame({"Common_Gene": ["A"], "ER": [0.8]})

        result = _configure_rnalocate_augmentation(
            gold,
            augment_er=True,
        )

        self.assertEqual(result["ER"].tolist(), [0.8])

    def test_apex_omm_specificity_filter_keeps_only_raw_specific_rows(self):
        raw = pd.DataFrame(
            {
                "Common_Gene": ["EDGE", "SHARED", "LOW"],
                "Source": ["test"] * 3,
                "Nucleus_log2FC": [0.0, 0.0, 0.0],
                "Cytosol_log2FC": [0.25, 0.26, 0.10],
                "ERM_log2FC": [0.20, 0.30, 0.10],
                "OMM_log2FC": [0.75, 0.75, 0.74],
            }
        )

        filtered, audit = _prepare_apex_gold_reference(
            raw,
            omm_min_log2fc=0.75,
            omm_min_delta=0.50,
        )
        legacy, _ = _prepare_apex_gold_reference(
            raw,
            omm_specificity_filter=False,
            omm_min_log2fc=0.75,
            omm_min_delta=0.50,
        )

        filtered = filtered.set_index("Common_Gene")
        legacy = legacy.set_index("Common_Gene")
        audit = audit.set_index("Common_Gene")
        self.assertTrue(audit.loc["EDGE", "omm_specific_pass"])
        self.assertGreater(filtered.loc["EDGE", "OMM"], 0.0)
        self.assertTrue(audit.loc["SHARED", "omm_shared"])
        self.assertEqual(filtered.loc["SHARED", "OMM"], 0.0)
        self.assertFalse(audit.loc["LOW", "omm_shared"])
        self.assertEqual(filtered.loc["LOW", "OMM"], 0.0)
        pd.testing.assert_series_equal(
            filtered.loc["SHARED"].drop("OMM"),
            legacy.loc["SHARED"].drop("OMM"),
        )

    def test_bundled_apex_omm_filter_regression_counts(self):
        gold, _, audit = _load_default_gold_tables(
            return_omm_audit=True,
        )

        self.assertEqual(int(audit["omm_basal_pass"].sum()), 1158)
        self.assertEqual(int(audit["omm_specific_pass"].sum()), 258)
        self.assertEqual(int(audit["omm_shared"].sum()), 900)
        self.assertEqual(
            int(
                audit.loc[
                    audit["omm_specific_pass"],
                    "Common_Gene",
                ]
                .dropna()
                .astype(str)
                .nunique()
            ),
            254,
        )
        self.assertEqual(int(gold["OMM"].gt(0).sum()), 254)


    def test_bool_parser_handles_h5ad_style_string_categories(self):
        values = pd.Series(
            pd.Categorical(["False", "True", None], categories=["False", "True"])
        )

        result = _coerce_bool_series(values, default=True)

        self.assertEqual(result.tolist(), [False, True, True])


    def test_intermediate_anndata_save_is_atomic_and_readable(self):
        adata = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        adata.obs["mixed_object"] = pd.Series(
            ["ER", pd.NA],
            index=adata.obs_names,
            dtype="object",
        )
        original = adata.obs["mixed_object"].copy()

        with tempfile.TemporaryDirectory() as out_dir, _local_config():
            _save_adata(adata, out_dir, "annotation_checkpoint")
            path = os.path.join(
                out_dir,
                "intermediate",
                "annotation_checkpoint.h5ad",
            )
            self.assertTrue(os.path.exists(path))
            reloaded = ad.read_h5ad(path)

        self.assertEqual(reloaded.shape, adata.shape)
        self.assertIn("mixed_object", reloaded.obs.columns)
        pd.testing.assert_series_equal(
            adata.obs["mixed_object"],
            original,
        )

    def test_resume_build_uses_aligned_final_table_in_lite_mode(self):
        adata1 = ad.AnnData(
            np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "cell_id": ["c1", "c2"],
                    "metaspot": ["m1", "m2"],
                    "log_n": np.log1p([2, 1]),
                },
                index=["m1", "m2"],
            ),
        )
        adata1.obsm["X_lens"] = np.zeros((2, 3), dtype=np.float32)
        adata2 = ad.AnnData(
            np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["d1", "d2"]),
        )
        final_data = pd.DataFrame(
            {
                "cell_id": ["c1", "c1", "c2"],
                "metaspot": ["m1", "m1", "m2"],
                "meta-domain": ["d1", "d1", "d2"],
                "subcellular_annotation": ["ER", "ER", "Nucleus"],
            }
        )

        with tempfile.TemporaryDirectory() as out_dir, _local_config():
            intermediate = os.path.join(out_dir, "intermediate")
            os.makedirs(intermediate, exist_ok=True)
            adata1.write_h5ad(
                os.path.join(intermediate, "module5_adata1.h5ad")
            )
            adata2.write_h5ad(
                os.path.join(intermediate, "module6_adata2.h5ad")
            )
            final_data.to_parquet(
                os.path.join(out_dir, "final_data.parquet"),
                index=False,
            )

            resumed = _resume_build_outputs(out_dir)

        self.assertIsNotNone(resumed)
        resumed_adata1, resumed_adata2, resumed_final = resumed
        self.assertEqual(resumed_adata1.n_obs, 2)
        self.assertEqual(resumed_adata2.n_obs, 2)
        self.assertEqual(len(resumed_final), 3)


    def test_segmentation_features_are_limited_by_local_variance(self):
        matrix = np.array(
            [
                [0.0, 0.0, 0.0, 4.0],
                [1.0, 0.1, 0.0, 3.0],
                [2.0, 0.2, 0.0, 2.0],
                [3.0, 0.3, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        selected = _select_top_variance_features(matrix, 2)

        self.assertEqual(selected.shape, (4, 2))
        np.testing.assert_array_equal(selected[:, 0], matrix[:, 0])
        np.testing.assert_array_equal(selected[:, 1], matrix[:, 3])

    def test_spatial_connectivity_splits_disconnected_islands(self):
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.zeros(4, dtype=np.int32)

        connected = _enforce_spatially_connected_labels(
            coordinates,
            labels,
            connectivity_k=2,
            radius_quantile=0.9,
            radius_factor=1.1,
            smoothing_iterations=0,
            min_component_size=1,
        )

        self.assertEqual(len(np.unique(connected)), 2)
        self.assertEqual(connected[0], connected[1])
        self.assertEqual(connected[2], connected[3])
        self.assertNotEqual(connected[0], connected[2])

    def test_spatial_ward_quiets_known_disconnected_graph_warning(self):
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        features = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)

        with warnings.catch_warnings(record=True) as caught:
            connected = _enforce_spatially_connected_labels(
                coordinates,
                labels,
                feature_matrix=features,
                target_n_clusters=2,
                connectivity_k=1,
                radius_quantile=0.9,
                radius_factor=1.1,
                smoothing_iterations=0,
                min_component_size=1,
            )

        disconnected_notices = [
            warning
            for warning in caught
            if "number of connected components of the connectivity matrix"
            in str(warning.message)
        ]
        self.assertEqual(disconnected_notices, [])
        self.assertEqual(connected[0], connected[1])
        self.assertEqual(connected[2], connected[3])
        self.assertNotEqual(connected[0], connected[2])

    def test_spatial_ward_early_stop_matches_full_tree_membership(self):
        rng = np.random.default_rng(91)
        coordinates = np.round(
            rng.uniform(0.0, 50.0, size=(600, 2)),
            2,
        ).astype(np.float32)
        features = rng.normal(size=(600, 12)).astype(np.float32)
        labels = np.zeros(600, dtype=np.int32)
        real_clustering = metaspots_module.AgglomerativeClustering

        def _full_tree_clustering(*args, **kwargs):
            kwargs["compute_full_tree"] = True
            return real_clustering(*args, **kwargs)

        with mock.patch.object(
            metaspots_module,
            "AgglomerativeClustering",
            side_effect=_full_tree_clustering,
        ):
            full_tree = _enforce_spatially_connected_labels(
                coordinates,
                labels,
                feature_matrix=features,
                target_n_clusters=30,
                connectivity_k=8,
                min_component_size=5,
            )
        early_stop = _enforce_spatially_connected_labels(
            coordinates,
            labels,
            feature_matrix=features,
            target_n_clusters=30,
            connectivity_k=8,
            min_component_size=5,
        )

        np.testing.assert_array_equal(early_stop, full_tree)

    def test_spatial_connectivity_merges_small_component_to_neighbor(self):
        coordinates = np.column_stack(
            (np.arange(6, dtype=np.float32), np.zeros(6, dtype=np.float32))
        )
        labels = np.asarray([0, 0, 1, 0, 0, 0], dtype=np.int32)

        connected = _enforce_spatially_connected_labels(
            coordinates,
            labels,
            connectivity_k=2,
            radius_quantile=0.9,
            radius_factor=1.1,
            smoothing_iterations=0,
            min_component_size=2,
        )

        self.assertIn(connected[2], {connected[1], connected[3]})
        self.assertLessEqual(len(np.unique(connected)), 2)

    def test_target_metaspot_count_uses_unique_sites_and_bounds(self):
        coordinates = np.repeat(
            np.column_stack(
                (np.arange(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32))
            ),
            repeats=3,
            axis=0,
        )

        self.assertEqual(_target_metaspot_count(coordinates, 100, 2, 100), 10)
        self.assertEqual(_target_metaspot_count(coordinates, 500, 5, 100), 5)
        self.assertEqual(_target_metaspot_count(coordinates, 5, 2, 20), 20)

    def test_segmentation_audit_vectorizes_requested_metaspot_counts(self):
        frame = pd.DataFrame(
            {
                "cell_id": ["A"] * 5 + ["B"] * 3 + ["C"] * 8,
                "metaspot": (
                    ["A:0", "A:0", "A:0", "A:1", "A:1"]
                    + ["B:0"] * 3
                    + ["C:0"] * 4
                    + ["C:1"] * 4
                ),
                "x_location": [
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    np.nan,
                    np.nan,
                    np.inf,
                    np.nan,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                ],
                "y_location": np.zeros(16, dtype=np.float32),
                "in_nucleus": [
                    True,
                    False,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                ],
            }
        )
        original_eq = pd.Series.eq

        def reject_full_cell_scan(series, other, *args, **kwargs):
            if series.name == "cell_id" and len(series) == len(frame):
                raise AssertionError("audit rescanned the full cell_id column")
            return original_eq(series, other, *args, **kwargs)

        with mock.patch.object(pd.Series, "eq", new=reject_full_cell_scan):
            audit = _metaspot_segmentation_audit(
                frame,
                target_sites_per_metaspot=2,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=3,
            )

        expected = {
            cell_id: _target_metaspot_count(
                group[["x_location", "y_location"]].to_numpy(
                    np.float32,
                    copy=False,
                ),
                target_sites_per_metaspot=2,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=3,
            )
            for cell_id, group in frame.groupby("cell_id", sort=False)
        }
        actual = audit.set_index("cell_id")["requested_metaspots"].to_dict()

        self.assertEqual(actual, expected)
        self.assertEqual(actual, {"A": 2, "B": 1, "C": 3})
        self.assertEqual(
            audit.set_index("cell_id")["n_unique_sites"].to_dict(),
            {"A": 4, "B": 2, "C": 8},
        )

    def test_partition_target_allocation_preserves_cell_budget(self):
        nucleus = np.column_stack(
            (
                np.arange(30, dtype=np.float32),
                np.zeros(30, dtype=np.float32),
            )
        )
        cytoplasm = np.column_stack(
            (
                np.arange(70, dtype=np.float32),
                np.ones(70, dtype=np.float32),
            )
        )

        allocated = _allocate_partition_target_counts(
            [nucleus, cytoplasm],
            target_sites_per_metaspot=10,
            min_metaspots_per_cell=1,
            max_metaspots_per_cell=8,
        )

        self.assertEqual(allocated, [2, 6])
        self.assertEqual(sum(allocated), 8)

    def test_module3_partition_prevents_nuclear_mixed_metaspots(self):
        n_per_side = 40
        frame = pd.DataFrame(
            {
                "cell_id": ["Cell1"] * (2 * n_per_side),
                "feature_name": ["GeneA", "GeneB"] * n_per_side,
                "x_location": np.r_[
                    np.linspace(0, 19, n_per_side),
                    np.linspace(20, 39, n_per_side),
                ].astype(np.float32),
                "y_location": np.zeros(2 * n_per_side, dtype=np.float32),
                "in_nucleus": np.r_[
                    np.ones(n_per_side, dtype=bool),
                    np.zeros(n_per_side, dtype=bool),
                ],
                "r_signed": np.r_[
                    -np.ones(n_per_side, dtype=np.float32),
                    np.ones(n_per_side, dtype=np.float32),
                ],
                "d_cell_norm": np.full(
                    2 * n_per_side,
                    0.5,
                    dtype=np.float32,
                ),
                "d_nuc_norm": np.full(
                    2 * n_per_side,
                    0.5,
                    dtype=np.float32,
                ),
            }
        )
        clustered = per_cell_clustering(
            frame,
            ["r_signed", "d_cell_norm", "d_nuc_norm"],
            min_segment_points=5,
            sequential_cells=10,
            spatial_connectivity_k=4,
            spatial_min_component_size=2,
            target_sites_per_metaspot=5,
            min_metaspots_per_cell=2,
            max_metaspots_per_cell=8,
        )
        mixed = (
            clustered.groupby("metaspot", observed=True)["in_nucleus"]
            .nunique()
            .gt(1)
        )
        self.assertFalse(mixed.any())
        self.assertLessEqual(clustered["metaspot"].nunique(), 8)

        audit = _metaspot_segmentation_audit(
            clustered,
            target_sites_per_metaspot=5,
            min_metaspots_per_cell=2,
            max_metaspots_per_cell=8,
        )
        self.assertEqual(
            int(audit["nucleus_mixed_metaspots"].sum()),
            0,
        )
        self.assertEqual(
            float(audit["nucleus_mixed_transcript_fraction"].max()),
            0.0,
        )

    def test_module4_cache_alignment_rejects_stale_metaspot_numbering(self):
        point_df = pd.DataFrame(
            {
                "metaspot": ["cell:0"] * 3 + ["cell:1"] * 3,
                "feature_name": ["A", "A", "B", "A", "B", "B"],
                "r_signed": [-0.8, -0.7, -0.6, 0.5, 0.6, 0.7],
                "d_cell_norm": [0.2] * 6,
                "d_nuc_norm": [0.8] * 6,
                "x_location": np.arange(6, dtype=np.float32),
                "y_location": np.zeros(6, dtype=np.float32),
            }
        )
        cached = build_adata1(point_df)

        aligned, reason = _validate_adata1_point_df_alignment(cached, point_df)
        self.assertTrue(aligned, reason)

        renumbered = point_df.copy()
        renumbered.loc[0, "metaspot"] = "cell:1"
        aligned, reason = _validate_adata1_point_df_alignment(
            cached,
            renumbered,
        )
        self.assertFalse(aligned)
        self.assertIn("transcript counts differ", reason)

        changed_geometry = point_df.copy()
        changed_geometry.loc[
            changed_geometry["metaspot"].eq("cell:0"),
            "r_signed",
        ] = 0.9
        aligned, reason = _validate_adata1_point_df_alignment(
            cached,
            changed_geometry,
        )
        self.assertFalse(aligned)
        self.assertIn("r_mean differs", reason)

    def test_module4_vectorized_statistics_match_reference_formulas(self):
        point_df = pd.DataFrame(
            {
                "metaspot": ["cell:0"] * 5 + ["cell:1"] * 4,
                "feature_name": ["A", "B", "A", "C", "B"] * 1
                + ["A", "C", "C", "B"],
                "r_signed": [-0.8, np.nan, -0.2, 0.1, 0.5, 0.2, 0.4, 0.7, 0.9],
                "d_cell_norm": np.linspace(0.1, 0.9, 9),
                "d_nuc_norm": np.linspace(0.9, 0.1, 9),
                "x_location": [0.0, np.nan, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
                "y_location": [0.0, 1.0, 0.0, 2.0, 1.0, 5.0, 5.5, np.nan, 7.0],
            }
        )

        adata1 = build_adata1(point_df)

        for metaspot, group in point_df.groupby("metaspot", sort=True):
            r_values = group["r_signed"].to_numpy(np.float32)
            expected_quantiles = np.nanpercentile(
                r_values,
                [10, 50, 90],
            )
            np.testing.assert_allclose(
                adata1.obs.loc[
                    metaspot,
                    ["r_p10", "r_p50", "r_p90"],
                ].to_numpy(dtype=float),
                expected_quantiles,
                rtol=0.0,
                atol=1e-6,
            )
            self.assertAlmostEqual(
                float(adata1.obs.loc[metaspot, "r_mean"]),
                float(np.nanmean(r_values)),
                places=6,
            )
            self.assertAlmostEqual(
                float(adata1.obs.loc[metaspot, "r_std"]),
                float(np.nanstd(r_values)),
                places=6,
            )

            x_values = group["x_location"].to_numpy(np.float32)
            y_values = group["y_location"].to_numpy(np.float32)
            centered_x = x_values - np.nanmean(x_values)
            centered_y = y_values - np.nanmean(y_values)
            finite = np.isfinite(centered_x) & np.isfinite(centered_y)
            denominator = max(int(finite.sum()) - 1, 1)
            var_x = float(
                np.square(centered_x[finite]).sum() / denominator
            )
            var_y = float(
                np.square(centered_y[finite]).sum() / denominator
            )
            covariance = float(
                (centered_x[finite] * centered_y[finite]).sum()
                / denominator
            )
            trace = max(var_x + var_y, 0.0)
            discriminant = max(
                trace * trace
                - 4.0 * (var_x * var_y - covariance * covariance),
                0.0,
            )
            root = np.sqrt(discriminant)
            lambda_1 = max(0.5 * (trace + root), 1e-8)
            lambda_2 = max(0.5 * (trace - root), 1e-8)
            np.testing.assert_allclose(
                float(adata1.obs.loc[metaspot, "elongation"]),
                lambda_1 / lambda_2,
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                float(adata1.obs.loc[metaspot, "spread"]),
                np.sqrt(trace),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_resume_fingerprint_tracks_applied_tuning_parameters(self):
        cfg = {
            "module1": {"use_input_nucleus_flags": False},
            "module2": {"idf_mode": "log"},
            "module3": {
                "n_segmentation_features": 12,
                "spatial_compactness": 2.0,
                "target_sites_per_metaspot": 100,
            },
            "module5": {"metadomain_q": 0.1, "lens_max_epochs": 17},
            "annotation": {
                "method": "evidence",
                "evidence_axis_q": 0.2,
                "apex_omm_specificity_filter": True,
                "apex_omm_min_delta": 0.50,
            },
        }

        self.assertEqual(module_config_subset(cfg, "m2")["module2.idf_mode"], "log")
        self.assertFalse(
            module_config_subset(cfg, "m1")[
                "module1.use_input_nucleus_flags"
            ]
        )
        self.assertEqual(
            module_config_subset(cfg, "m3")["module3.n_segmentation_features"],
            12,
        )
        self.assertEqual(
            module_config_subset(cfg, "m3")["module3.spatial_compactness"],
            2.0,
        )
        self.assertEqual(
            module_config_subset(cfg, "m3")[
                "module3.target_sites_per_metaspot"
            ],
            100,
        )
        self.assertEqual(
            module_config_subset(cfg, "m5")["module5.metadomain_q"],
            0.1,
        )
        annotate_cfg = stage_config_subset(cfg, "annotate")
        self.assertEqual(annotate_cfg["annotation.method"], "evidence")
        self.assertEqual(annotate_cfg["annotation.evidence_axis_q"], 0.2)
        self.assertTrue(
            annotate_cfg["annotation.apex_omm_specificity_filter"]
        )
        self.assertEqual(
            annotate_cfg["annotation.apex_omm_min_delta"],
            0.50,
        )

    def test_streaming_model_preserves_local_expression_without_global_columns(self):
        frame = _synthetic_transcripts()
        with _local_config():
            output, _, model = build_composition_embeddings(
                frame.copy(),
                frame,
                frame["cell_id"].unique().tolist(),
            )

        self.assertIsInstance(model, _LocalCompositionModel)
        self.assertFalse(any(c.startswith("comp_emb") for c in output.columns))

        embedding, density = model.transform_cell(frame[frame["cell_id"] == "0"])
        separation = np.linalg.norm(
            embedding[:15].mean(axis=0) - embedding[25:].mean(axis=0)
        )
        self.assertGreater(separation, 0.1)
        self.assertTrue(np.isfinite(density).all())

    def test_module3_consumes_streaming_expression_model(self):
        frame = _synthetic_transcripts()
        real_import = __import__

        def reject_hdbscan(name, *args, **kwargs):
            if name == "hdbscan":
                raise AssertionError(
                    "canonical spatial clustering must not import hdbscan"
                )
            return real_import(name, *args, **kwargs)

        with _local_config(), mock.patch(
            "builtins.__import__",
            side_effect=reject_hdbscan,
        ):
            point_df, gmm_cols, model = build_composition_embeddings(
                frame.copy(),
                frame,
                frame["cell_id"].unique().tolist(),
            )
            clustered = per_cell_clustering(
                point_df,
                gmm_cols,
                composition_model=model,
                min_segment_points=5,
                sequential_cells=10,
                spatial_min_component_size=2,
                spatial_compactness=1.0,
                target_sites_per_metaspot=10,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=4,
            )

        self.assertIn("metaspot_in_cell", clustered.columns)
        self.assertIn("metaspot", clustered.columns)
        self.assertGreaterEqual(
            clustered.loc[
                clustered["cell_id"] == "0",
                "metaspot_in_cell",
            ].nunique(),
            2,
        )
        self.assertFalse(any(c.startswith("comp_emb") for c in clustered.columns))

        adata1 = build_adata1(clustered)
        self.assertEqual(adata1.n_obs, clustered["metaspot"].nunique())

    def test_module3_worker_does_not_capture_full_point_table(self):
        frame = _synthetic_transcripts()
        captured = {}

        class CaptureParallel:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __call__(self, tasks):
                worker, args, _ = next(iter(tasks))
                captured["freevars"] = set(worker.__code__.co_freevars)
                cell_id, group = args[0]
                return [
                    (
                        cell_id,
                        pd.Series(
                            np.zeros(len(group), dtype=np.int32),
                            index=group.index,
                        ),
                    )
                ]

        with _local_config(), mock.patch(
            "joblib.Parallel",
            CaptureParallel,
        ):
            per_cell_clustering(
                frame,
                ["r_signed", "d_cell_norm", "d_nuc_norm"],
                sequential_cells=0,
                small_backend_cells=0,
                parallel_chunk_cells=16,
                parallel_n_jobs=2,
                target_sites_per_metaspot=10,
                min_metaspots_per_cell=2,
                max_metaspots_per_cell=4,
            )

        self.assertNotIn("point_df", captured["freevars"])


if __name__ == "__main__":
    unittest.main()


def test_relative_type_expression_flags_genes_of_the_other_cell_type():
    from cellscoper.spillover import relative_type_expression

    rng = np.random.default_rng(0)
    rows = []
    for cell in range(40):
        own, other = ("A", "B") if cell < 20 else ("B", "A")
        for prefix, n in ((own, 95), (other, 5)):
            rows += [(f"c{cell}", f"{prefix}{g}") for g in rng.integers(0, 5, n)]
    frame = pd.DataFrame(rows, columns=["cell_id", "feature_name"])
    relative = relative_type_expression(frame, n_types=2)
    is_own = frame["feature_name"].str[0] == np.where(
        frame["cell_id"].str[1:].astype(int) < 20, "A", "B"
    )
    assert (relative[is_own] > 0.9).all()
    assert (relative[~is_own] < 0.2).all()
