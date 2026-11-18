import os
import tempfile
from contextlib import contextmanager

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from cellscoper.config import fingerprint_module, fingerprint_stage
from cellscoper.keys import LENS_REP
from cellscoper.metadomains import run_module5_simple
from cellscoper.outputs import _write_output_guide
from cellscoper.spatial_gene_embedding import GENE_EMBEDDING_REP


@contextmanager
def _config_file(config):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    previous = os.environ.get("CELLSCOPER_CONFIG")
    os.environ["CELLSCOPER_CONFIG"] = path
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop("CELLSCOPER_CONFIG", None)
        else:
            os.environ["CELLSCOPER_CONFIG"] = previous
        os.unlink(path)


def _metaspots(n_cells=30, per_side=8, seed=0):
    """MetaSpots with precomputed GE / LENS blocks (no training) and a nuclear side (r_p50 < 0)."""
    rng = np.random.default_rng(seed)
    n = n_cells * per_side * 2
    names = [f"c{c}:{s}" for c in range(n_cells) for s in range(per_side * 2)]
    obs = pd.DataFrame(
        {"cell_id": np.repeat([f"c{c}" for c in range(n_cells)], per_side * 2),
         "r_p50": np.tile(np.r_[-np.ones(per_side), np.ones(per_side)], n_cells)},
        index=names,
    )
    adata = ad.AnnData(X=np.zeros((n, 2), np.float32), obs=obs)
    adata.obsm[GENE_EMBEDDING_REP] = rng.normal(size=(n, 12)).astype(np.float32)
    adata.obsm[LENS_REP] = rng.normal(size=(n, 6)).astype(np.float32)
    points = pd.DataFrame({"metaspot": names, "cell_id": obs["cell_id"].to_numpy()})
    return adata, points


def test_module5_metadomains_come_from_lens_discovery(monkeypatch):
    import cellscoper.metadomain_discovery as discovery

    adata, points = _metaspots()
    adata.uns["lens"] = {"null_spots": np.arange(10)}
    found = np.where(np.arange(adata.n_obs) % 3 == 0, -1, np.arange(adata.n_obs) % 2)
    calls = {}

    def fake(adata1, point_df, *, q, seed):
        calls.update(q=q, seed=seed)
        return found, np.where(found >= 0, 1.5, 0.0), {"metadomain": np.array([0, 1])}

    monkeypatch.setattr(discovery, "discover_for_adata", fake)
    with _config_file({"module5": {"metadomain_q": 0.1, "random_seed": 7}}):
        adata, diag = run_module5_simple(adata, points)
    assert calls == {"q": 0.1, "seed": 7}
    assert adata.obs["meta-domain"].astype(str).tolist() == found.astype(str).tolist()  # background = "-1"
    assert int(diag["n_metadomains"].iloc[0]) == 2
    assert adata.uns["module5_metadomain_clustering"]["clustering_method"] == "lens_hdbscan_discovery"


def test_module5_needs_the_lens_null():
    adata, points = _metaspots()
    adata.uns["lens"] = {"null_spots": np.zeros(0)}
    with _config_file({"module5": {}}):
        try:
            run_module5_simple(adata, points)
        except ValueError as exc:
            assert "lens_null_cells" in str(exc)
        else:
            raise AssertionError("expected a ValueError")


def test_metadomain_parameters_change_module5_fingerprint_only():
    base = {"module5": {"metadomain_q": 0.2}, "module3": {"target_sites_per_metaspot": 15}}
    changed = {"module5": {"metadomain_q": 0.1}, "module3": {"target_sites_per_metaspot": 15}}
    assert fingerprint_module(base, "m5") != fingerprint_module(changed, "m5")
    assert fingerprint_module(base, "m3") == fingerprint_module(changed, "m3")
    assert fingerprint_stage(base, "build_anndata") != fingerprint_stage(changed, "build_anndata")


def test_output_guide_lists_primary_products_and_annotation_columns(tmp_path):
    counts = {"n_transcripts": 100, "n_cells": 5, "n_metaspots": 20, "n_metadomains": 4}
    with _config_file({"io": {"intermediate_dirname": "intermediate_results"}}):
        guide = _write_output_guide(str(tmp_path), counts=counts, annotation_enabled=True)
        text = open(guide, encoding="utf-8").read()
        assert "Primary data products" in text
        assert "subcellular_annotation_validated" in text and "evidence_annotation_report.json" in text
        assert "cell_structure_features.parquet" in text and "meta-domain" in text
        assert "dmgi" not in text.lower()
        csv_guide = _write_output_guide(str(tmp_path), counts=counts, annotation_enabled=False,
                                        primary_data_product="final_data.csv")
        assert "`final_data.csv`: transcript-level production result" in open(csv_guide, encoding="utf-8").read()
