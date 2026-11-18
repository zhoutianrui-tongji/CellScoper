import numpy as np
import pandas as pd

from cellscoper.evidence_annotation import run_evidence_annotation
from cellscoper.spatial_gene_embedding import metaspot_gene_embedding
from cellscoper.structure_axes import AxisData, false_share_curve, read_weight, smooth_neighbours, threshold_for


def _cells(n_cells=40, with_patch=True, seed=0):
    """Cytoplasm only unless noted; in the first half of the cells 10 marker genes sit in one corner patch."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        for nuc, x0 in ((False, 0.0), (True, 20.0)):
            n = 1200
            xy = rng.uniform(0, 10, (n, 2)) + [x0, 0]
            genes = rng.integers(0, 60, n)
            if with_patch and c < n_cells // 2 and not nuc:
                patch = (xy[:, 0] < 3) & (xy[:, 1] < 3)
                genes[patch] = rng.integers(60, 70, patch.sum())
            blocks = np.floor(xy[:, 0] / 2).astype(int) * 10 + np.floor(xy[:, 1] / 2).astype(int)
            rows.append(pd.DataFrame({"cell_id": f"c{c}", "feature_name": [f"G{g}" for g in genes], "x_location": xy[:, 0], "y_location": xy[:, 1],
                                      "in_nucleus": nuc, "metaspot": [f"c{c}_{int(nuc)}_{b}" for b in blocks], "meta-domain": "d0"}))
    return pd.concat(rows, ignore_index=True)


def _reference():
    genes = [f"G{g}" for g in range(60, 70)] + [f"G{g}" for g in range(0, 8)]
    table = pd.DataFrame(0.0, index=genes, columns=["ER", "OMM", "Nuclear_Speckle", "Nucleolus", "Lamina", "Nuclear_Pore"])
    table.loc[[f"G{g}" for g in range(60, 70)], "ER"] = 1.0
    table.loc[[f"G{g}" for g in range(0, 8)], "OMM"] = 1.0
    return table


def _annotate(df, q=0.2):
    names = pd.Index(np.sort(df.metaspot.unique()))
    ge = {}
    block = metaspot_gene_embedding(df, names, scale="multiscale", dim=6, min_gene_count=20, gene_vectors=ge)
    return run_evidence_annotation(df, _reference(), {"evidence_label_rule": "axis", "evidence_axis_source": "ge", "evidence_axis_q": q, "evidence_validation_folds": 2, "evidence_min_resolution": -1.0,
                                                      "evidence_bootstrap": 100}, {"spot_names": names, "ge": ge, "ge_block": block})


def test_false_share_is_low_for_a_real_tail_and_a_threshold_needs_it():
    rng = np.random.default_rng(0)
    null = rng.normal(0, 1, 50000)
    real = np.r_[rng.normal(0, 1, 45000), rng.normal(4, 0.5, 5000)]
    tau = threshold_for(real, null, 0.05)
    assert tau is not None and 2 < tau < 4
    grid, curve = false_share_curve(real, null)
    assert (np.diff(curve) <= 1e-12).all()  # a q-value never rises with the score
    assert threshold_for(rng.normal(0, 1, 50000), null, 0.05) is None  # no signal, no threshold


def test_structure_is_labelled_where_it_is_and_only_there():
    df = _cells()
    table, report = _annotate(df)
    t = table.set_index("metaspot")
    er = t.index[t.subcellular_annotation == "ER"]
    assert report["structures"]["ER"]["detected"] and len(er) > 0
    cell = er.str.split("_").str[0].str[1:].astype(int)
    assert (cell < 20).mean() > 0.9  # no seats: cells without the patch stay without ER
    block = er.str.split("_").str[2].astype(int)
    assert ((block // 10 <= 1) & (block % 10 <= 1)).mean() > 0.8  # inside the corner
    assert report["structures"]["ER"]["level"] == "axis" and t.subcellular_annotation_placed.isin(["ER"]).sum() >= len(np.unique(cell))


def test_shuffled_genes_give_almost_no_labels():
    df = _cells(with_patch=False)
    table, report = _annotate(df)
    assert (table.subcellular_annotation == "ER").mean() < 0.02
    assert (table.subcellular_annotation == "OMM").mean() < 0.02


def test_smoothing_follows_transcripts_not_metaspots():
    assert smooth_neighbours(10) == 5 and smooth_neighbours(50) == 1 and smooth_neighbours(2) == 25


def test_read_weight_is_finite_for_a_degenerate_null_readout():
    weight = read_weight(np.array([0.0, 1.0, 2.0]), np.ones(3))
    assert np.isfinite(weight) and weight > 0


def test_axis_data_shuffle_keeps_each_cell_sides_genes():
    df = _cells(n_cells=3)
    data = AxisData(df, np.sort(df.metaspot.unique()))
    rows = np.arange(len(df))
    shuffled = data.shuffled_genes(rows)
    assert not np.array_equal(shuffled, data.gene_codes)
    key = data.tx_cell * 2 + data.part.astype(int)
    for k in np.unique(key):
        assert sorted(shuffled[key == k]) == sorted(data.gene_codes[key == k])


def test_mitochondrially_encoded_genes_become_the_mitochondrion_markers():
    df = _cells(n_cells=10)
    df["feature_name"] = df["feature_name"].replace({f"G{i}": f"MT-ND{i + 1}" for i in range(4)})
    _, report = _annotate(df)
    assert report["reference_marker_genes"]["OMM"] == 4


def _annotate_radial(df, radial_of_spot, bands):
    names = pd.Index(np.sort(df.metaspot.unique()))
    ge = {}
    block = metaspot_gene_embedding(df, names, scale="multiscale", dim=6, min_gene_count=20, gene_vectors=ge)
    return run_evidence_annotation(df, _reference(), {"evidence_label_rule": "axis", "evidence_axis_source": "ge", "evidence_axis_q": 0.2, "evidence_validation_folds": 2,
                                                      "evidence_min_resolution": -1.0, "evidence_bootstrap": 100, "evidence_axis_radial_bands": bands},
                                   {"spot_names": names, "ge": ge, "ge_block": block, "radial": radial_of_spot(names)})


def test_centre_by_band_removes_band_means_only():
    from cellscoper.structure_axes import centre_by_band
    band = np.array([0, 0, 1, 1, 2])
    out = centre_by_band(np.array([1.0, 3.0, 10.0, 12.0, 5.0]), band)
    np.testing.assert_allclose(out, [-1, 1, -1, 1, 0])


def test_radial_bands_keep_a_patch_unrelated_to_position_and_remove_a_shared_radial_trend():
    df = _cells()
    rng = np.random.default_rng(1)
    random_r = lambda names: rng.uniform(0, 1, len(names))
    # 径向位置与斑块无关：斑块仍被找到，且仍在角上
    t = _annotate_radial(df, random_r, 5)[0].set_index("metaspot")
    er = t.index[t.subcellular_annotation == "ER"]
    block = er.str.split("_").str[2].astype(int)
    assert len(er) > 0 and ((block // 10 <= 1) & (block % 10 <= 1)).mean() > 0.8
    # 每个细胞同一径向位置（x < 2 的一条）都富集标志基因：这是群体共有的趋势，去趋势后这条带上的 ER 标签大幅减少
    strip = _cells(with_patch=False, seed=2)
    on = ~strip.in_nucleus & (strip.x_location < 2) & (np.random.default_rng(3).random(len(strip)) < 0.5)
    strip.loc[on, "feature_name"] = [f"G{g}" for g in np.random.default_rng(4).integers(60, 70, on.sum())]
    x_r = lambda names: (pd.Index(names).str.split("_").str[2].astype(int) // 10).to_numpy() / 4.0
    in_strip = lambda t: t.metaspot.str.split("_").str[2].astype(int) // 10 == 0
    base = _annotate(strip)[0]
    flat = _annotate_radial(strip, x_r, 5)[0]
    n_base = ((base.subcellular_annotation == "ER") & in_strip(base)).sum()
    n_flat = ((flat.subcellular_annotation == "ER") & in_strip(flat)).sum()
    assert n_base > 50 and n_flat < 0.2 * n_base


def test_text_codes_match_factorize_of_text_for_categorical_columns():
    from cellscoper.structure_axes import text_codes
    raw = pd.Series(["b", "a", None, "b", "c", "a", None, "10"], dtype=object)
    for col in (raw, raw.astype("category"), pd.Series(pd.Categorical([3, 1, 3, 2, 1]))):
        want_codes, want_text = pd.factorize(col.astype(str).to_numpy())
        codes, text = text_codes(col)
        np.testing.assert_array_equal(codes, want_codes)
        np.testing.assert_array_equal(text, np.asarray(want_text).astype(str))


def test_axis_data_is_the_same_for_categorical_and_text_columns():
    df = _cells(n_cells=6)
    names = np.sort(df.metaspot.unique())
    a = AxisData(df, names)
    cat = df.copy()
    for c in ("cell_id", "metaspot", "feature_name"):
        cat[c] = cat[c].astype("category")
    b = AxisData(cat, names)
    for attr in ("codes", "tx_cell", "gene_codes", "gene_names", "size", "spot_key"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    rows = np.arange(len(df))
    np.testing.assert_array_equal(a.shuffled_genes(rows), b.shuffled_genes(rows))
