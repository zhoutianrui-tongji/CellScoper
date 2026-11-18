import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellscoper import run_figures
from cellscoper.spillover import relative_type_expression


def _toy_run(rng, n_cells=8, spots_per_cell=5, tx_per_spot=6):
    rows = []
    for c in range(n_cells):
        for s in range(spots_per_cell):
            for _ in range(tx_per_spot):
                rows.append((f"c{c}", f"c{c}_{s}", *rng.uniform(0, 10, 2), rng.choice(["ER", "Nucleus", "Cytosol"]), s % 3 - 1))
    final = pd.DataFrame(rows, columns=["cell_id", "metaspot", "x_location", "y_location", "subcellular_annotation", "meta-domain"])
    obs = final.groupby("metaspot", sort=False).agg(cell_id=("cell_id", "first"), n=("cell_id", "size"))
    obs["log_n"] = np.log1p(obs.pop("n"))
    S = len(obs)
    adata = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, (S, 30)).astype(np.float32)), obs=obs,
                       var=pd.DataFrame(index=[f"G{i}" for i in range(30)]))
    adata.obs["meta-domain"] = (np.arange(S) % 3 - 1).astype(str)
    adata.obs["subcellular_broad_compartment"] = np.where(np.arange(S) % 2, "Nucleus", "Cytosol")
    adata.obsm["X_lens"] = rng.normal(size=(S, 4)).astype(np.float32)
    genes = np.array([f"G{i}" for i in range(30)])
    adata.uns["gene_embedding"] = {"vectors": rng.normal(size=(30, 6)).astype(np.float32), "genes": genes}
    adata.uns["lens"] = {"null_spots": np.arange(10), "null_z": rng.normal(size=(10, 4)), "heldout_spot_gain": rng.normal(size=S)}
    adata.uns["metadomain_discovery"] = {"metadomain": np.array([0]), "side": np.array(["both"]), "metaspots": np.array([12]),
                                         "nucleus_share": np.array([0.4]), "real_above": np.array([0.2]),
                                         "null_above": np.array([0.05]), "top_genes": np.array(["G1,G2,G3"])}
    report = {"reference_marker_genes": {"ER": 2, "Nucleolus": 1},  # 报告里只存个数
              "structures": {"ER": {"supported": True, "level": "axis", "levels": {"axis": {
                  "marker_enrichment": 1.3, "ci_low": 1.2, "ci_high": 1.4}}},
                  "Nuclear Pore": {"supported": False, "level": "axis", "levels": {}}},
              "label_metaspot_fraction": {"ER": 0.2, "Nucleus": 0.5, "Cytosol": 0.3},
              "lens_count": {"structures": {"ER": {"auroc": 0.8, "null_mean": 0.5, "null_sd": 0.02, "pass": True},
                                            "Nuclear Pore": {"auroc": None}}}}
    adata.uns["evidence_annotation"] = json.dumps(report)
    features = pd.DataFrame({"cell_id": np.repeat(final.cell_id.unique(), 2), "in_nucleus": [True, False] * n_cells,
                             "kind": "structure", "feature": ["Nucleolus", "ER"] * n_cells,
                             "n_transcripts": rng.integers(10, 100, 2 * n_cells), "transcript_share": rng.random(2 * n_cells)})
    bounds = pd.DataFrame({"cell_id": np.repeat(final.cell_id.unique(), 4), "vertex_x": np.tile([0, 10, 10, 0], n_cells),
                           "vertex_y": np.tile([0, 0, 10, 10], n_cells)})
    return adata, final, features, bounds


def test_run_figures_write_every_figure(tmp_path):
    adata, final, features, bounds = _toy_run(np.random.default_rng(0))
    run_figures.write_run_figures(str(tmp_path), adata, final, cell_features=features, cell_bounds=bounds, nucleus_bounds=bounds,
                                  markers={"ER": ["G1", "G2", "G4"], "Nucleolus": ["G3", "G5", "G6"]})
    written = sorted(p.stem for p in (tmp_path / run_figures.FIGURES_DIR).glob("*.png"))
    assert written == ["02_metaspots", "03_gene_embedding", "05_lens_signal", "06_metadomains",
                       "07_annotation_validation", "08_structure_share_vs_depth", "09_example_cells"]


def test_run_figures_skip_missing_inputs_without_error(tmp_path):
    adata, _, _, _ = _toy_run(np.random.default_rng(1))
    adata.uns["metadomain_discovery"] = {"metadomain": np.array([])}  # 没找到 MetaDomain：画说明面板
    del adata.uns["lens"], adata.uns["evidence_annotation"]
    run_figures.write_run_figures(str(tmp_path), adata)
    written = sorted(p.stem for p in (tmp_path / run_figures.FIGURES_DIR).glob("*.png"))
    assert written == ["02_metaspots", "06_metadomains"]


def test_foreign_removal_figure_and_cell_types_match_default_output(tmp_path):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"cell_id": np.repeat([f"c{i}" for i in range(20)], 30),
                          "feature_name": rng.choice([f"g{i}" for i in range(8)], 600)})
    relative, cell_codes, cell_type = relative_type_expression(frame, n_types=3, return_cell_types=True)
    assert np.array_equal(relative, relative_type_expression(frame, n_types=3))
    assert len(cell_type) == 20 and cell_codes.max() == 19
    run_figures.plot_foreign_removal(str(tmp_path), cell_codes, cell_type, relative <= 0.2, 0.2)
    assert (tmp_path / run_figures.FIGURES_DIR / "01_foreign_transcripts.png").is_file()


def test_observed_domain_genes_ranks_by_enrichment_against_same_side_background():
    # 背景：基因 0/1/2 各 100 条；MetaDomain 0：基因 2 是 300 条（富集），基因 0/1 各 100 条；基因 3 只有 5 条（低于门槛）
    counts = sp.csr_matrix(np.array([[100, 100, 100, 0], [100, 100, 300, 5]], np.float32))
    out = run_figures.observed_domain_genes(counts, np.array([-1, 0]), None, ["A", "B", "C", "D"], n_top=2, min_count=150)
    assert out[0][0][0] == "C" and "D" not in [g for g, _ in out[0]]
    assert abs(out[0][0][1] - 301 / (505 / 3 + 1)) < 1e-6  # 期望 = 背景占比 1/3 × MetaDomain 总数 505


def test_text_overlap_check_finds_overlapping_labels():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 2))
    ax = fig.subplots()
    ax.text(0.5, 0.5, "first label")
    ax.text(0.5, 0.5, "second label")
    assert len(run_figures.text_overlaps(fig)) >= 1


def test_structure_direction_stats_sees_gene_signal_along_the_marker_direction():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(50, 4))
    W[:5] += np.array([3.0, 0, 0, 0])  # 基因 0–4 = ER marker，方向约为第 1 维
    null_z = rng.normal(size=(400, 4))
    real = null_z * 0.3 + rng.normal(size=(400, 4))
    real[:, 0] += rng.normal(scale=3.0, size=400)  # 真实细胞在 ER 方向上多出基因信号
    lens = {"null_spots": np.arange(400), "null_z": null_z, "gene_loadings": W,
            "gene_names": np.array([f"G{i}" for i in range(50)])}
    out = run_figures.structure_direction_stats(real, lens, {"ER": [f"G{i}" for i in range(5)]}, np.zeros(400, bool))
    assert len(out) == 1 and out[0][0] == "ER" and out[0][1] > 1.5 and out[0][2] < 0.5
