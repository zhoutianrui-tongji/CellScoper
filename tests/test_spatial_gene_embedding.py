import unittest

import numpy as np
import pandas as pd

from cellscoper.spatial_gene_embedding import (
    MULTISCALE_NN,
    fit_gene_embedding,
    metaspot_gene_embedding,
    resolve_radius,
)


def _two_zone_cells(n_cells=40, per_zone=120, seed=0):
    """Each cell: genes A0-A4 in a left zone, B0-B4 in a right zone, C0-C4 everywhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        for zone, genes, x0 in (("left", "A", 0.0), ("right", "B", 6.0)):
            for k in range(per_zone):
                rows.append((f"cell{c}", f"{genes}{k % 5}", x0 + rng.uniform(0, 3), rng.uniform(0, 3), zone))
        for k in range(per_zone):
            rows.append((f"cell{c}", f"C{k % 5}", rng.uniform(0, 9), rng.uniform(0, 3), "mixed"))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name", "x_location", "y_location", "zone"])
    df["x_location"] += 100.0 * (np.arange(len(df)) // (3 * per_zone))  # keep cells apart in space
    return df


class SpatialGeneEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.df = _two_zone_cells()
        self.cells = pd.factorize(self.df["cell_id"])[0]
        self.codes, self.genes = pd.factorize(self.df["feature_name"])
        self.xy = self.df[["x_location", "y_location"]].to_numpy(float)

    def test_parallel_workers_give_identical_vectors(self):
        df = self.df.assign(metaspot=self.df["cell_id"] + ":" + self.df["zone"])
        names = pd.Index(df["metaspot"].unique())
        one = metaspot_gene_embedding(df, names, dim=4, min_gene_count=50, n_jobs=1)
        two = metaspot_gene_embedding(df, names, dim=4, min_gene_count=50, n_jobs=2)
        np.testing.assert_array_equal(one, two)

    def test_genes_sharing_a_zone_get_similar_vectors(self):
        kept, emb = fit_gene_embedding(self.cells, self.codes, self.xy, len(self.genes), dim=4, min_gene_count=50)
        names = self.genes[kept]
        unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = pd.DataFrame(unit @ unit.T, index=names, columns=names)
        a = [g for g in names if g.startswith("A")]
        b = [g for g in names if g.startswith("B")]
        within = sim.loc[a, a].to_numpy()[~np.eye(len(a), dtype=bool)].mean()
        across = sim.loc[a, b].to_numpy().mean()
        self.assertGreater(within, across + 0.3)

    def test_metaspot_matrix_is_aligned_and_empty_metaspots_are_zero(self):
        df = self.df.assign(metaspot=self.df["cell_id"] + "_" + self.df["zone"])
        names = pd.Index(sorted(df["metaspot"].unique()) + ["empty_spot"])
        matrix = metaspot_gene_embedding(df, names, dim=4, min_gene_count=50)
        self.assertEqual(matrix.shape, (len(names), 4))
        self.assertTrue(np.all(matrix[-1] == 0))
        self.assertTrue(np.isfinite(matrix).all())

    def test_multiscale_concatenates_one_block_per_scale(self):
        df = self.df.assign(metaspot=self.df["cell_id"] + "_" + self.df["zone"])
        names = pd.Index(sorted(df["metaspot"].unique()))
        matrix = metaspot_gene_embedding(df, names, dim=4, min_gene_count=50, scale="multiscale")
        self.assertEqual(matrix.shape, (len(names), 4 * len(MULTISCALE_NN)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_too_few_genes_is_an_error(self):
        with self.assertRaises(ValueError):
            fit_gene_embedding(self.cells, self.codes, self.xy, len(self.genes), min_gene_count=10**9)

    def test_auto_radii_do_not_depend_on_coordinate_units(self):
        df = self.df.assign(metaspot=self.df["cell_id"] + "_" + self.df["zone"])
        names = pd.Index(sorted(df["metaspot"].unique()))
        in_um = metaspot_gene_embedding(df, names, dim=4, min_gene_count=50)
        in_pixels = metaspot_gene_embedding(
            df.assign(x_location=df["x_location"] * 10, y_location=df["y_location"] * 10), names, dim=4, min_gene_count=50
        )
        np.testing.assert_allclose(in_um, in_pixels, atol=1e-4)

    def test_z_separates_layers_that_overlap_in_x_y(self):
        # Same x-y spread for A and B genes, but A sits 3 units below B: only z can tell them apart.
        df = self.df[self.df["zone"] != "mixed"].copy()
        df["x_location"] = df["x_location"] % 100 % 6 + 100.0 * df["cell_id"].str[4:].astype(int)
        df["z_location"] = np.where(df["zone"] == "left", 0.0, 3.0)
        df["metaspot"] = df["cell_id"] + "_" + df["zone"]
        names = pd.Index(sorted(df["metaspot"].unique()))
        gap = lambda m: np.linalg.norm(m[names.str.endswith("left")].mean(0) - m[names.str.endswith("right")].mean(0))
        with_z = metaspot_gene_embedding(df, names, dim=4, min_gene_count=50, pair_radius=1.0, neighbour_radius=1.5)
        without_z = metaspot_gene_embedding(df.drop(columns="z_location"), names, dim=4, min_gene_count=50,
                                            pair_radius=1.0, neighbour_radius=1.5)
        self.assertGreater(gap(with_z), 5 * gap(without_z))

    def test_numeric_radius_is_used_as_given(self):
        self.assertEqual(resolve_radius(2.5, nearest=0.1, multiple=15), 2.5)
        self.assertAlmostEqual(resolve_radius("auto", nearest=0.1, multiple=15), 1.5)
        self.assertAlmostEqual(resolve_radius(None, nearest=0.1, multiple=15), 1.5)


def test_gene_vectors_output_matches_the_metaspot_embedding():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"cell_id": rng.integers(0, 10, n).astype(str), "feature_name": rng.integers(0, 12, n).astype(str),
                       "x_location": rng.uniform(0, 5, n), "y_location": rng.uniform(0, 5, n), "metaspot": rng.integers(0, 40, n).astype(str)})
    names = pd.Index(sorted(df.metaspot.unique()))
    out = {}
    with_vectors = metaspot_gene_embedding(df, names, scale="multiscale", dim=4, min_gene_count=5, gene_vectors=out)
    np.testing.assert_array_equal(with_vectors, metaspot_gene_embedding(df, names, scale="multiscale", dim=4, min_gene_count=5))
    assert out["vectors"].shape == (len(out["genes"]), with_vectors.shape[1])
    assert out["scale_dims"].sum() == with_vectors.shape[1] and len(out["neighbour_radii"]) == len(out["scale_dims"])


if __name__ == "__main__":
    unittest.main()


def test_many_variant_metaspot_vectors_match_the_one_by_one_computation():
    import numpy as np
    from scipy import sparse
    from scipy.spatial import cKDTree
    from cellscoper.spatial_gene_embedding import _cell_groups, _metaspot_vectors_many

    def old_vectors(cells, gene_codes, xy, part, codes, kept, embedding, n_genes, n_spots, radius):  # 2026-11-12 前的实现
        lookup = np.zeros((n_genes, embedding.shape[1]), dtype=np.float32); lookup[kept] = embedding
        has_vector = np.zeros(n_genes, dtype=bool); has_vector[kept] = True
        sums = np.zeros((n_spots, embedding.shape[1]), dtype=np.float64)
        for idx in _cell_groups(cells):
            x, local, p, spot = xy[idx], gene_codes[idx], part[idx], codes[idx]
            out = np.zeros((len(local), lookup.shape[1]), dtype=np.float32)
            pairs = cKDTree(x).query_pairs(radius, output_type="ndarray")
            if len(pairs):
                i = np.r_[pairs[:, 0], pairs[:, 1]]; j = np.r_[pairs[:, 1], pairs[:, 0]]; ok = has_vector[local[j]]
                adj = sparse.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (i[ok], j[ok])), shape=(len(local), len(local)))
                deg = np.asarray(adj.sum(axis=1)).ravel(); vec = (adj @ lookup[local]) / np.maximum(deg, 1)[:, None]
                for v in np.unique(p):
                    out[p == v] = vec[p == v] - vec[p == v].mean(axis=0)
            rows = np.flatnonzero(spot >= 0); spots, col = np.unique(spot[rows], return_inverse=True)
            sums[spots] += sparse.csr_matrix((np.ones(rows.size), (col, rows)), shape=(len(spots), len(spot))) @ out.astype(np.float64)
        n = np.bincount(codes[codes >= 0], minlength=n_spots)
        return (sums / np.maximum(n, 1)[:, None]).astype(np.float32)

    rng = np.random.default_rng(0)
    n, G = 8000, 40
    cells = rng.integers(0, 30, n); cells[0] = 99                       # 只有一个转录本的细胞
    xy = rng.uniform(0, 6, (n, 3)); part = rng.random(n) < 0.4
    codes = cells * 10 + rng.integers(0, 10, n); codes[rng.random(n) < 0.02] = -1   # 部分行不属于任何 MetaSpot
    codes = np.unique(codes, return_inverse=True)[1] - (np.unique(codes)[0] == -1)
    genes = rng.integers(0, G, n); shuffled = rng.permutation(genes)
    kept = np.arange(0, G, 2); emb = rng.normal(size=(len(kept), 6)).astype(np.float32)
    drop = rng.random(len(kept)) < 0.3
    variants = [(genes, kept, emb), (shuffled, kept, emb), (genes, kept[~drop], emb[~drop]), (shuffled, kept[~drop], emb[~drop])]
    S = codes.max() + 1
    for radius in (0.5, 1.5):
        many = _metaspot_vectors_many(cells, [v[0] for v in variants], xy, part, codes, [v[1] for v in variants], [v[2] for v in variants], G, S, radius, 2)
        for (g, k, e), got in zip(variants, many):
            np.testing.assert_array_equal(got, old_vectors(cells, g, xy, part, codes, k, e, G, S, radius))
