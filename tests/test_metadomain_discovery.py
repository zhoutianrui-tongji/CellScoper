import numpy as np

import pandas as pd

from cellscoper.metadomain_discovery import (consensus_directions, discover_metadomains, domain_gene_profiles, marker_auc,
                                             nucleus_boundary_check)


def _spots(n_cells=300, per_cell=40, dim=8, planted=True, seed=0):
    """MetaSpots of one side: isotropic noise, and (planted) two directions carried by a fifth of the MetaSpots each."""
    rng = np.random.default_rng(seed)
    n = n_cells * per_cell
    z = rng.normal(size=(n, dim)).astype(np.float32)
    null = rng.normal(size=(n, dim)).astype(np.float32)
    truth = np.full(n, -1)
    if planted:
        kind = rng.random(n)
        truth[kind < 0.2] = 0
        truth[(kind >= 0.2) & (kind < 0.4)] = 1
        z[truth == 0, 0] += 6.0
        z[truth == 1, 1] += 6.0
    cell = np.repeat(np.arange(n_cells), per_cell)
    centroid = rng.normal(size=(n, 3))
    depth = np.repeat(rng.integers(200, 2000, n_cells), per_cell).astype(float)
    return z, null, truth, cell, centroid, depth


def _run(z, null, cell, centroid, depth):
    n = len(z)
    nuclear = np.zeros(n, bool)
    return discover_metadomains(z, null, np.arange(n), nuclear, cell * 2, centroid, depth, cell, smooth_k=1, max_fit=5000)


def test_planted_directions_are_found_without_a_fixed_number():
    z, null, truth, cell, centroid, depth = _spots()
    domain, score, table, sides, zs = _run(z, null, cell, centroid, depth)
    assert len(table) == 2
    # each planted group is mostly one MetaDomain, and background stays background
    for t in (0, 1):
        found = domain[truth == t]
        assert np.mean(found == np.bincount(found[found >= 0]).argmax()) > 0.8
    assert np.mean(domain[truth == -1] >= 0) < 0.1
    assert (table.repro_half1 > 0.8).all() and (table.repro_half2 > 0.8).all()


def test_no_metadomain_when_real_matches_the_null():
    z, null, truth, cell, centroid, depth = _spots(planted=False)
    domain, score, table, sides, zs = _run(z, null, cell, centroid, depth)
    assert len(table) == 0 and (domain == -1).all()


def test_decoder_profile_names_the_planted_genes():
    z, null, truth, cell, centroid, depth = _spots()
    domain, score, table, sides, zs = _run(z, null, cell, centroid, depth)
    W = np.zeros((20, z.shape[1]))
    W[:5, 0] = 1.0   # genes 0-4 read direction 0
    W[5:10, 1] = 1.0
    prof = domain_gene_profiles(zs, domain, W, len(table))
    genes = [f"G{i}" for i in range(20)]
    best = max(range(len(table)), key=lambda k: marker_auc(prof[k], genes, genes[:5]))
    assert marker_auc(prof[best], genes, genes[:5]) > 0.9


def test_one_direction_on_both_sides_is_one_metadomain_unless_split():
    """The same planted direction in nuclear and cytoplasmic MetaSpots: one MetaDomain spanning both sides when clustered
    together, a mirror pair when each side is clustered alone."""
    z, null, truth, cell, centroid, depth = _spots(n_cells=400)
    rng = np.random.default_rng(1)
    nuclear = rng.random(len(z)) < 0.5
    z[truth == 1, 1] -= 6.0     # keep one planted direction only
    truth[truth == 1] = -1
    key = cell * 2 + nuclear
    joint = discover_metadomains(z, null, np.arange(len(z)), nuclear, key, centroid, depth, cell, smooth_k=1, max_fit=5000)
    split = discover_metadomains(z, null, np.arange(len(z)), nuclear, key, centroid, depth, cell, smooth_k=1, max_fit=5000,
                                 split_sides=True)
    assert len(joint[2]) == 1 and joint[2].side.iloc[0] == "both" and 0.3 < joint[2].nucleus_share.iloc[0] < 0.7
    assert len(split[2]) == 2 and set(split[2].side) == {"nucleus", "cytoplasm"}


def test_nucleus_boundary_check_flags_only_domains_across_the_outline():
    genes = [f"G{i}" for i in range(50)]
    lo = pd.Series(np.linspace(-1, 1, 50), index=genes)          # G49 most nuclear
    nuclear_like = lo.to_numpy() + np.random.default_rng(0).normal(0, .1, 50)
    profiles = np.vstack([nuclear_like, nuclear_like])
    nuclear = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    domain = np.full(200, -1)
    domain[:100] = 0                  # nucleus only: speckle-like
    domain[50:150] = 1                # half inside, half outside the outline
    out = nucleus_boundary_check(profiles, genes, lo, domain, nuclear)
    assert out.nucleus_axis_r.min() > 0.9
    assert not out.nucleus_boundary_flag.iloc[0] and out.nucleus_boundary_flag.iloc[1]


def test_consensus_keeps_directions_found_in_most_restarts():
    e = np.eye(4)
    one = np.ones(1)
    runs = [(e[[0, 1]], np.r_[.9, .9], np.r_[.9, .9]), (e[[1, 0]], np.r_[.9, .9], np.r_[.9, .9]),
            (e[[0, 2]], np.r_[.9, .9], np.r_[.9, .9]), (np.zeros((0, 4)), one[:0], one[:0]), (e[[0]], one * .9, one * .9)]
    directions, report = consensus_directions(runs, 4, min_repro=0.8)
    # e0 in 4 of 5 restarts, e1 in 2 (below half), e2 in 1
    assert directions.shape == (1, 4) and np.allclose(directions[0], e[0])
    assert report[0]["support"] == 0.8


def test_metadomains_do_not_depend_on_the_seed():
    z, null, truth, cell, centroid, depth = _spots(seed=3)
    n = len(z)
    counts = set()
    for seed in (0, 1, 2):
        domain = discover_metadomains(z, null, np.arange(n), np.zeros(n, bool), cell * 2, centroid, depth, cell,
                                      smooth_k=1, max_fit=5000, seed=seed)[0]
        counts.add(int(domain.max()) + 1)
    assert counts == {2}


def test_gene_nuclear_log_odds_matches_the_text_groupby_exactly():
    import numpy as np
    import pandas as pd
    from cellscoper.metadomain_discovery import gene_nuclear_log_odds
    rng = np.random.default_rng(0)
    n = 20000
    cell = pd.Series(rng.choice([f"c{i}" for i in range(50)], n))
    gene = pd.Series(rng.choice(["gA", "GA", "gb", "MT-X", "z1"], n))
    y = rng.random(n) < 0.4
    # 原实现：逐行大写、按文字分组
    t = pd.DataFrame({"cell": cell.to_numpy(), "gene": gene.astype(str).str.upper().to_numpy(), "y": y})
    t["p"] = t.groupby("cell").y.transform("mean")
    g = t.groupby("gene").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    want = np.log((g.obs + 1) / (g.exp + 1)) - np.log((g.n - g.obs + 1) / (g.n - g.exp + 1))
    for c, gg in ((cell, gene), (cell.astype("category"), gene.astype("category")), (cell.to_numpy(), gene.to_numpy())):
        pd.testing.assert_series_equal(gene_nuclear_log_odds(c, gg, y), want, check_exact=True)
