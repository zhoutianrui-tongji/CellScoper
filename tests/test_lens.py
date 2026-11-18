import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from cellscoper.lens import Z_DIM, lens_embedding


def test_lens_embedding_is_aligned_and_centred_per_cell_side():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(12):
        for s in range(6):
            nuc = s < 2
            centre = rng.uniform(0, 10, 2)
            genes = rng.choice(10, 25)  # ~180 transcripts per gene: above the 50-transcript vocabulary cut
            for g in genes:
                x, y = centre + rng.normal(0, 0.3, 2)
                rows.append((f"c{c}", f"g{g}", x, y, nuc, f"c{c}_{s}", rng.random(), rng.random()))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(["empty"] + sorted(df.metaspot.unique()))
    Z, diag = lens_embedding(df, names, max_epochs=2, min_gene_count=5, device="cpu")
    assert Z.shape == (len(names), Z_DIM)
    assert np.all(Z[0] == 0)  # MetaSpot without transcripts
    side = df.groupby("metaspot").in_nucleus.first().reindex(names[1:]).to_numpy()
    cell = names[1:].str.split("_").str[0]
    means = pd.DataFrame(Z[1:]).groupby([cell, side]).mean().to_numpy()
    assert np.abs(means).max() < 1e-5
    assert np.isfinite(diag["best_heldout_gain"]) and 1 <= diag["epochs"] <= 2
    assert diag["heldout_spot_gain"].shape == (len(names),) and np.isfinite(diag["heldout_spot_gain"]).any()


def test_lens_sampled_training_uses_fixed_spots_per_epoch():
    rng = np.random.default_rng(1)
    rows = [(f"c{c}", f"g{g}", *rng.uniform(0, 10, 2), s < 2, f"c{c}_{s}", rng.random(), rng.random())
            for c in range(12) for s in range(6) for g in rng.choice(10, 25)]
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    # 20 MetaSpots per epoch: the 60-unit cap raises max_epochs from 1 to 3
    Z, diag = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu", spots_per_epoch=20,
                                     patience_units=40, max_units=60)
    assert Z.shape == (len(names), Z_DIM) and np.isfinite(Z).all()
    assert diag["spots_per_epoch"] == 20 and diag["epochs"] == 3


def test_lens_large_data_paths_match_shapes(monkeypatch):
    import cellscoper.lens as lens_module
    rng = np.random.default_rng(2)
    rows = [(f"c{c}", f"g{g}", *rng.uniform(0, 10, 2), s < 2, f"c{c}_{s}", rng.random(), rng.random())
            for c in range(12) for s in range(6) for g in rng.choice(10, 25)]
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    small, _ = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu")
    # NMF fitted on 10 of the 24 cell sides and projected; host tables (forced)
    monkeypatch.setattr(lens_module, "NMF_FIT_MAX", 10)
    monkeypatch.setattr(lens_module, "GPU_TABLE_MAX_BYTES", 0)
    Z, diag = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu")
    assert Z.shape == small.shape and np.isfinite(Z).all() and np.isfinite(diag["best_heldout_gain"])


def test_lens_context_is_same_side_and_never_the_metaspot_itself():
    import cellscoper.lens as lens_module
    rng = np.random.default_rng(3)
    xyz = np.c_[rng.uniform(0, 10, (200, 2)), np.zeros(200)]
    key = np.repeat([0, 1], 100)  # two cell sides
    local = np.repeat(np.arange(20), 10)  # 20 MetaSpots of 10 transcripts, 10 per side
    spot_key = np.repeat([0, 1], 10)
    centroid = np.vstack([xyz[local == s].mean(0) for s in range(20)])
    ctx = lens_module._context(xyz, key, local, spot_key, centroid, 1e9)
    full = ctx >= 0
    assert (full.sum(1) == lens_module.MAX_LEN).all()  # 90 other transcripts per side, the 32 nearest kept
    assert (key[ctx] == key[np.arange(20) * 10][:, None])[full].all()  # same cell side
    assert (local[ctx] != np.arange(20)[:, None])[full].all()  # never the MetaSpot's own transcripts


def test_lens_null_encoding_leaves_the_real_embedding_unchanged():
    rng = np.random.default_rng(1)
    rows = []
    for c in range(12):
        for s in range(6):
            centre = rng.uniform(0, 10, 2)
            for g in rng.choice(10, 25):
                x, y = centre + rng.normal(0, 0.3, 2)
                rows.append((f"c{c}", f"g{g}", x, y, s < 2, f"c{c}_{s}", rng.random(), rng.random()))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    Z0, d0 = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu")
    Z1, d1 = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu", null_cells=4)
    assert len(d0["null_spots"]) == 0 and d0["null_z"].shape == (0, Z_DIM)
    np.testing.assert_allclose(Z0, Z1, atol=1e-5)
    assert d1["null_z"].shape == (4 * 6, Z_DIM) and len(set(names[d1["null_spots"]].str.split("_").str[0])) == 4
    assert not np.allclose(d1["null_z"], Z1[d1["null_spots"]])  # genes were shuffled


def test_lens_decoder_loadings_cover_the_vocabulary():
    rng = np.random.default_rng(6)
    rows = []
    for c in range(12):
        for s in range(6):
            centre = rng.uniform(0, 10, 2)
            for g in rng.choice(10, 25):
                x, y = centre + rng.normal(0, 0.3, 2)
                rows.append((f"c{c}", f"g{g}", x, y, s < 2, f"c{c}_{s}", rng.random(), rng.random()))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    z, diag = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu")
    assert z.shape == (len(names), Z_DIM) and np.isfinite(z).all()
    assert diag["gene_loadings"].shape == (10, Z_DIM) and np.isfinite(diag["gene_loadings"]).all()
    assert len(diag["gene_names"]) == 10


def test_context_matches_the_per_metaspot_loop():
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    from cellscoper.lens import MAX_LEN, _context
    rng = np.random.default_rng(0)
    n = 6000
    key = rng.integers(0, 40, n)                 # 细胞侧
    key[:1] = 99                                 # 只有一个转录本的细胞侧
    xyz = rng.uniform(0, 5, (n, 3))
    local = np.full(n, -1)
    spot_key, centroid = [], []
    for k in np.unique(key):                      # 每侧切成若干 MetaSpot
        rows = np.flatnonzero(key == k)
        parts = np.array_split(rows, max(1, len(rows) // 12))
        for p in parts:
            local[p] = len(spot_key)
            spot_key.append(k)
            centroid.append(xyz[p].mean(0))
    spot_key, centroid = np.asarray(spot_key), np.asarray(centroid)

    def loop_version(radius):  # 原实现
        S = len(centroid)
        ctx = np.full((S, MAX_LEN), -1, np.int64)
        order = np.argsort(key, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1, len(order)]
        spots_by_key = pd.Series(np.arange(S)).groupby(spot_key).indices
        for a, b in zip(bounds[:-1], bounds[1:]):
            members = order[a:b]
            qs = spots_by_key.get(key[members[0]])
            dist, nb = cKDTree(xyz[members]).query(centroid[qs], k=min(len(members), MAX_LEN + 32), distance_upper_bound=radius)
            dist, nb = np.atleast_2d(dist), np.atleast_2d(nb)
            if len(members) == 1:
                dist, nb = dist.reshape(-1, 1), nb.reshape(-1, 1)
            for r, sp in enumerate(qs):
                cand = members[nb[r][np.isfinite(dist[r])]]
                cand = cand[local[cand] != sp][:MAX_LEN]
                ctx[sp, :len(cand)] = cand
        return ctx

    for radius in (0.3, 1.0, 10.0):
        np.testing.assert_array_equal(_context(xyz, key, local, spot_key, centroid, radius), loop_version(radius))


def test_lens_own_background_trains_and_rejects_unknown_background():
    rng = np.random.default_rng(3)
    rows = [(f"c{c}", f"g{g}", *rng.uniform(0, 10, 2), s < 2, f"c{c}_{s}", rng.random(), rng.random())
            for c in range(12) for s in range(6) for g in rng.choice(10, 25)]
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    Z, diag = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu", background="own")
    assert Z.shape == (len(names), Z_DIM) and np.isfinite(Z).all()
    assert diag["background"] == "own" and diag["background_weights_own_lowrank_global"] == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="background"):
        lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu", background="global")


def test_lens_without_absolute_position_ignores_radial_columns():
    rng = np.random.default_rng(4)
    rows = [(f"c{c}", f"g{g}", *rng.uniform(0, 10, 2), s < 2, f"c{c}_{s}", rng.random(), rng.random())
            for c in range(12) for s in range(6) for g in rng.choice(10, 25)]
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    a, da = lens_embedding(df, names, max_epochs=1, min_gene_count=5, device="cpu", background="own", absolute_position=False)
    shuffled = df.assign(d_nuc_norm=rng.random(len(df)), d_cell_norm=rng.random(len(df)))
    b, _ = lens_embedding(shuffled, names, max_epochs=1, min_gene_count=5, device="cpu", background="own", absolute_position=False)
    assert da["absolute_position"] is False
    np.testing.assert_allclose(a, b, atol=1e-5)  # radial columns are not read


def test_lens_training_history_is_recorded_and_saved(tmp_path):
    from cellscoper.pipeline import _save_lens_training

    rng = np.random.default_rng(1)
    rows = [(f"c{c}", f"g{g}", *rng.uniform(0, 10, 2), s < 2, f"c{c}_{s}", rng.random(), rng.random())
            for c in range(12) for s in range(6) for g in rng.choice(10, 25)]
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "in_nucleus", "metaspot",
                                     "d_nuc_norm", "d_cell_norm"])
    names = pd.Index(sorted(df.metaspot.unique()))
    _, diag = lens_embedding(df, names, max_epochs=3, min_gene_count=5, device="cpu")
    hist = diag["history"]
    assert diag["stop_reason"] == "max_epochs" and diag["epochs"] == 3
    assert hist[diag["best_epoch"] - 1]["heldout_gain"] == diag["best_heldout_gain"]
    assert all(np.isfinite(h["train_nll"]) for h in hist) and hist[-1]["seconds"] >= hist[0]["seconds"]
    assert diag["n_train_metaspots"] + diag["n_val_metaspots"] == len(names)

    uns = {k: v for k, v in diag.items() if k != "history"}
    for col in ("heldout_gain", "train_nll", "seconds"):
        uns[f"{col}_history"] = np.asarray([h[col] for h in hist])
    _save_lens_training(uns, str(tmp_path))
    out = pd.read_csv(next(tmp_path.rglob("module5_lens_training.csv")))
    assert list(out.epoch) == [1, 2, 3] and out.best.sum() == 1
    assert int(out.loc[out.best, "epoch"].iloc[0]) == diag["best_epoch"]
