import numpy as np
import pandas as pd


def _cells(n_cells=60, n=300, clustered=True, seed=0):
    """Cytoplasm transcripts of genes A/B (weight 1) and C-F (weight 0); A/B either packed in one corner or anywhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        genes = rng.choice(list("ABCDEF"), n)
        xyz = rng.random((n, 3)) * 10
        if clustered:
            ab = np.isin(genes, ["A", "B"])
            xyz[ab] = rng.random((ab.sum(), 3)) * 3     # A/B packed in a 3 um corner
        for g, (x, y, z) in zip(genes, xyz):
            rows.append((f"c{c}", False, g, x, y, z, f"c{c}_{int(x // 2)}_{int(y // 2)}", "ER" if g in "AB" else "Cytosol"))
    return pd.DataFrame(rows, columns=["cell_id", "in_nucleus", "feature_name", "x_location", "y_location", "z_location",
                                       "metaspot", "subcellular_annotation"])


def test_pair_concentration_is_one_at_random_and_above_one_when_packed_and_unbiased_by_thinning():
    from cellscoper.cell_features import pair_concentration, pooled_concentration
    w = pd.DataFrame({"ER": [1.0, 1.0, 0, 0, 0, 0]}, index=list("ABCDEF"))
    rand = pair_concentration(_cells(clustered=False), w, radius=1.5)
    packed_points = _cells(clustered=True)
    packed = pair_concentration(packed_points, w, radius=1.5)
    groups = pd.Series("all", index=packed.cell_id.unique())
    r_rand = pooled_concentration(rand, groups).concentration.iloc[0]
    r_full = pooled_concentration(packed, groups).concentration.iloc[0]
    assert abs(r_rand - 1) < 0.05 and r_full > 2
    thin = packed_points.sample(frac=0.5, random_state=1)
    r_thin = pooled_concentration(pair_concentration(thin, w, radius=1.5), groups).concentration.iloc[0]
    assert abs(r_thin - r_full) / (r_full - 1) < 0.1


def test_pair_expectation_matches_brute_force_shuffling():
    from cellscoper.cell_features import _pair_parts_one_side
    rng = np.random.default_rng(3)
    xyz, code = rng.random((80, 3)) * 4, rng.integers(0, 5, 80)
    w = rng.random((5, 1))
    n_pairs, obs, exp = _pair_parts_one_side(xyz, code, w, 1.0)
    sims = [_pair_parts_one_side(xyz, rng.permutation(code), w, 1.0)[1][0] for _ in range(3000)]
    assert abs(np.mean(sims) - exp[0]) / exp[0] < 0.02


def test_structure_feature_table_shares_and_metadomain_rows():
    from cellscoper.cell_features import structure_feature_table
    pts = _cells(n_cells=10)
    pts["meta-domain"] = np.where(pts.feature_name == "C", "0", "-1")
    table = structure_feature_table(pts, structure_weights=pd.DataFrame({"ER": [1.0, 1.0, 0, 0, 0, 0]}, index=list("ABCDEF")),
                                    radius=1.5, domain_col="meta-domain",
                                    domain_weights=pd.DataFrame({"MD0": [0, 0, 1.0, 0, 0, 0]}, index=list("ABCDEF")))
    assert set(table.kind) == {"structure", "metadomain"} and len(table) == 20
    er = table[table.feature == "ER"]
    assert np.allclose(er.transcript_share, 1 / 3, atol=0.08) and (er.concentration > 1.5).all()
    assert table.loc[table.feature == "MD0", "metaspot_share"].between(0, 1).all()
