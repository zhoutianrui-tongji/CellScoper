"""LENS counting labels (cellscoper/lens_count.py): gate, threshold, quota placement, annotation option."""
import numpy as np
import pandas as pd
import pytest

from cellscoper.lens_count import annotate_lens_count, otsu_threshold, place_by_quota, structure_gate


def _loadings(n_genes=400, n_markers=40, dim=16, signal=3.0, seed=0):
    """Markers share one direction in W when signal > 0; other genes are noise."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_genes, dim))
    W[:n_markers, 0] += signal
    return W


def test_gate_passes_shared_direction_and_rejects_random_markers():
    W = _loadings()
    totals = np.random.default_rng(1).integers(50, 5000, len(W)).astype(float)
    non_ref = np.ones(len(W), bool); non_ref[:40] = False
    good = structure_gate(W, np.arange(40), non_ref, totals, n_null=10)
    assert good["pass"] and good["auroc"] > 0.9 and good["z"] >= 3
    W0 = _loadings(signal=0.0)
    bad = structure_gate(W0, np.arange(40), non_ref, totals, n_null=10)
    assert not bad["pass"] and abs(bad["auroc"] - 0.5) < 0.15
    assert not structure_gate(W, np.arange(3), non_ref, totals)["pass"]  # fewer markers than folds


def test_otsu_threshold_splits_two_modes():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.1, 0.02, 500), rng.normal(0.6, 0.05, 200)]
    t = otsu_threshold(x)  # same as skimage.filters.threshold_otsu (checked 2026-11-17)
    assert (x[:500] < t).mean() > 0.9 and (x[500:] > t).all()
    assert otsu_threshold(np.full(5, 0.3)) == 0.3


def test_place_by_quota_fills_quota_in_rank_order_and_resolves_conflicts():
    group = np.array([0, 0, 0, 0, 1, 1])
    size = np.array([10, 10, 10, 10, 5, 5], float)
    score = {"A": np.array([4, 3, 2, 1, 2, 1], float), "B": np.array([4, 1, 3, 2, 1, 2], float)}
    quota = {"A": np.array([20.0, 5.0]), "B": np.array([10.0, 0.0])}
    lab = place_by_quota(group, size, score, quota)
    # spot 0 is first for both; A reaches it at 10/20 = .25 of its quota, B at 5/10 = .5 -> A
    assert lab[0] == 0
    # B continues down its ranking to spot 2; A's second spot is spot 1
    assert lab[2] == 1 and lab[1] == 0 and lab[3] == -1
    assert lab[4] == 0 and lab[5] == -1  # cell 1: B has no quota
    for k, s in enumerate(("A", "B")):
        got = np.bincount(group[lab == k], weights=size[lab == k], minlength=2)
        assert np.all(got <= quota[s] + 1e-9)


def _toy_cells(seed=0):
    """Two cell types; structure genes g0-g39 sit together in one corner of each cytoplasm. Type 'rich' has 3x more of
    them, so its counting labels must cover more of the cytoplasm."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(40):
        rich = c % 2 == 0
        for s in range(12):
            corner = s < (6 if rich else 2)
            for _ in range(10):
                # background genes as frequent as the structure genes, so expression-matched null sets exist
                g = rng.integers(0, 40) if corner and rng.random() < 0.8 else rng.integers(40, 160)
                rows.append((f"c{c}", f"G{g}", False, f"c{c}_{s}"))
            rows.append((f"c{c}", f"G{rng.integers(40, 160)}", True, f"c{c}_n{s}"))
    return pd.DataFrame(rows, columns=["cell_id", "feature_name", "in_nucleus", "metaspot"])


def test_annotate_lens_count_labels_more_where_the_structure_is_abundant():
    df = _toy_cells()
    genes = df.feature_name.str.upper().to_numpy(); inside = df.in_nucleus.to_numpy()
    spot_codes, spots = pd.factorize(df.metaspot); cell_codes, _ = pd.factorize(df.cell_id)
    first = np.full(len(spots), -1); first[spot_codes[::-1]] = np.arange(len(df))[::-1]
    spot_cell, spot_nuc = cell_codes[first], inside[first]
    lens = {"gene_names": np.array([f"G{i}" for i in range(400)]), "gene_loadings": _loadings(), "background": "own"}
    markers = {"ER": [f"G{i}" for i in range(40)], "OMM": [], "Nucleolus": [], "Lamina": [], "Nuclear Pore": [], "Nuclear Speckle": []}
    labels, rep = annotate_lens_count(genes, inside, spot_codes, spot_cell, spot_nuc,
                                      {True: ("Nucleolus", "Lamina", "Nuclear Pore", "Nuclear Speckle"), False: ("ER", "OMM")},
                                      markers, lens, {}, {True: "Nucleus", False: "Cytosol"}, seed=0)
    assert rep["supported_structures"] == ["ER"] and not rep["structures"]["OMM"]["pass"]
    tx = pd.Series(labels[spot_codes] == "ER")
    share = tx[~inside].groupby(df.cell_id[~inside].str[1:].astype(int) % 2 == 0).mean()
    assert share[True] > 2 * share[False]  # abundant type gets more labels
    assert set(labels[spot_nuc]) == {"Nucleus"}


def test_lens_count_needs_own_background():
    df = _toy_cells()
    spot_codes, spots = pd.factorize(df.metaspot)
    lens = {"gene_names": np.array([f"G{i}" for i in range(400)]), "gene_loadings": _loadings(), "background": "learned"}
    with pytest.raises(ValueError, match="lens_background: own"):
        annotate_lens_count(df.feature_name.to_numpy(), df.in_nucleus.to_numpy(), spot_codes, np.zeros(len(spots), int),
                            np.zeros(len(spots), bool), {False: ("ER",)}, {"ER": ["G0"]}, lens, {}, {True: "Nucleus", False: "Cytosol"})
