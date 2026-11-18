"""Modules 5-6: the MetaSpot representation (spatial gene embedding + LENS), MetaDomains found from LENS without
reference genes (HDBSCAN, metadomain_discovery.py) and the MetaDomain AnnData (adata2)."""

from typing import Any, Callable, Dict, Mapping, Optional
import logging as _logging
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import load_params_yaml
from .keys import LENS_REP, METASPOT_COL
from .runtime import _preferred_n_jobs
from .spatial_gene_embedding import GENE_EMBEDDING_REP, metaspot_gene_embedding


def _compute_gene_embedding(point_df: pd.DataFrame, metaspot_names, m5_config: Mapping[str, Any], uns: Optional[Any] = None) -> np.ndarray:
    """Spatial gene embedding per MetaSpot with the module5.gene_embedding_* settings; gene vectors go to uns['gene_embedding']."""
    m5 = m5_config or {}
    gene_vectors: Dict[str, Any] = {}
    embedding = metaspot_gene_embedding(
        point_df,
        metaspot_names,
        metaspot_col=METASPOT_COL,
        pair_radius=m5.get('gene_embedding_pair_radius', 'auto'),
        neighbour_radius=m5.get('gene_embedding_neighbour_radius', 'auto'),
        scale=m5.get('gene_embedding_scale', 'multiscale'),
        dim=int(m5.get('gene_embedding_dim', 32)),
        min_gene_count=int(m5.get('gene_embedding_min_gene_count', 100)),
        pseudo_count=float(m5.get('gene_embedding_pseudo_count', 5.0)),
        max_cells=int(m5.get('gene_embedding_max_cells', 20000)),
        seed=int(m5.get('random_seed', 42)),
        log=_logging.getLogger("cellscoper").info,
        n_jobs=_preferred_n_jobs(),
        gene_vectors=gene_vectors,
    )
    if uns is not None:
        uns["gene_embedding"] = gene_vectors
    return embedding




def run_module5_simple(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """Module 5: MetaSpot representation and MetaDomains.

    Representation: the spatial gene embedding (GE, ``X_gene_embedding``) and LENS (``X_lens``, masked-transcript model,
    lens.py). Annotation reads both (structure axes); MetaDomains use LENS.

    MetaDomains (``meta-domain``) are found from LENS without reference genes (metadomain_discovery.py): directions of
    z that real MetaSpots take far more often than gene-shuffled cells, found by density (HDBSCAN, nucleus and cytoplasm
    together) and kept when both halves of the cells show them; their number comes from the data. MetaSpots outside
    every MetaDomain are background (``-1``).
    """
    import time as _time

    from .metadomain_discovery import discover_for_adata

    _module_label = "Module 5 · MetaDomains"

    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 5 · {name}")
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {name}")
            except Exception:
                pass
        return _time.perf_counter()

    def _step_done(name: str, t0: float):
        if show_internal_progress:
            print(f"<< Done:  {name} in {_time.perf_counter() - t0:0.2f}s")

    _m5 = (load_params_yaml() or {}).get("module5", {}) or {}
    random_seed = int(_m5.get("random_seed", 42))
    metadomain_q = float(_m5.get("metadomain_q", 0.2))

    if GENE_EMBEDDING_REP not in adata1.obsm:
        t_ge = _time.perf_counter()
        adata1.obsm[GENE_EMBEDDING_REP] = _compute_gene_embedding(point_df, adata1.obs_names, _m5, adata1.uns)
        _step_done("Spatial gene embedding", t_ge)
    if LENS_REP not in adata1.obsm:
        from .lens import lens_embedding

        t_lens = _step_start("LENS (masked-transcript model)")
        adata1.obsm[LENS_REP], lens_diag = lens_embedding(
            point_df,
            adata1.obs_names,
            metaspot_col=METASPOT_COL,
            max_epochs=int(_m5.get("lens_max_epochs", 200)),
            spots_per_epoch=int(_m5.get("lens_spots_per_epoch") or 0),
            null_cells=int(_m5.get("lens_null_cells", 10000)),
            background=str(_m5.get("lens_background", "own")),
            absolute_position=bool(_m5.get("lens_absolute_position", False)),
            min_gene_count=int(_m5.get("gene_embedding_min_gene_count", 100)),
            max_cells=int(_m5.get("gene_embedding_max_cells", 20000)),
            seed=random_seed,
            log=_logging.getLogger("cellscoper").info,
            n_jobs=_preferred_n_jobs(),
        )
        adata1.uns["lens"] = {k: v for k, v in lens_diag.items() if k != "history"}
        for col in ("heldout_gain", "train_nll", "seconds"):
            adata1.uns["lens"][f"{col}_history"] = np.asarray([h[col] for h in lens_diag["history"]])
        _step_done("LENS (masked-transcript model)", t_lens)
    if "cell_id" not in adata1.obs.columns:
        owners = (
            point_df[[METASPOT_COL, "cell_id"]]
            .assign(**{METASPOT_COL: lambda frame: frame[METASPOT_COL].astype(str)})
            .drop_duplicates()
            .set_index(METASPOT_COL)["cell_id"]
            .astype(str)
        )
        if owners.index.duplicated().any():
            raise ValueError("A MetaSpot has more than one owning cell")
        adata1.obs = adata1.obs.copy()
        adata1.obs["cell_id"] = owners.reindex(adata1.obs_names.astype(str)).to_numpy()

    if len(adata1.uns["lens"].get("null_spots", ())) == 0:
        raise ValueError("MetaDomains need the gene-shuffled LENS null: set module5.lens_null_cells > 0")
    t_md = _step_start("MetaDomain discovery")
    domain, score, summary = discover_for_adata(adata1, point_df, q=metadomain_q, seed=random_seed)
    adata1.obs = adata1.obs.copy()
    adata1.obs["meta-domain"] = pd.Categorical(domain.astype(np.int64).astype(str))
    adata1.obs["meta-domain_score"] = score.astype(np.float32)
    adata1.uns["metadomain_discovery"] = summary
    n_domains = int(len(summary.get("metadomain", ())))
    adata1.uns["module5_metadomain_clustering"] = {
        "clustering_method": "lens_hdbscan_discovery",
        "random_seed": int(random_seed),
        "q": metadomain_q,
        "n_metaspots": int(adata1.n_obs),
        "n_metadomains": n_domains,
        "background_share": float(np.mean(domain < 0)),
    }
    _step_done("MetaDomain discovery", t_md)
    _logging.getLogger("cellscoper").info(
        "Module 5 · MetaDomains: %d found, %.1f%% of MetaSpots inside one", n_domains, 100.0 * float(np.mean(domain >= 0)))
    return adata1, pd.DataFrame({"n_metaspots": [adata1.n_obs], "n_metadomains": [n_domains],
                                 "background_share": [float(np.mean(domain < 0))]})


def build_adata2_from_adata1_and_pointdf(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    show_internal_progress: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ad.AnnData:
    import time as _time
    _module_label = "Module 6 · Aggregate Domains into Cluster Feature Table (adata2)"
    def _notify(desc: str):
        if progress_callback:
            try:
                progress_callback(f"{_module_label} >> Start: {desc}")
            except Exception:
                pass
    def _step_start(name: str):
        if show_internal_progress:
            print(f">> Start: Module 6 · {name}")
        _notify(name)
        return _time.perf_counter()
    def _step_done(name: str, t0: float):
        if show_internal_progress:
            dt = _time.perf_counter() - t0
            print(f"<< Done:  {name} in {dt:0.2f}s")
    # 1) normalize rows (TF/CPM-like)
    # Optional overall progress bar for M6 (fixed 5 steps)
    _m6_prog = None; _m6_task = None
    if show_internal_progress:
        try:
            from rich.console import Console as _Console
            from rich.progress import Progress as _Progress, SpinnerColumn as _Spin, BarColumn as _Bar, TextColumn as _Text, TimeElapsedColumn as _Time
            _m6_console = _Console(stderr=True)
            if _m6_console.is_terminal:
                _m6_prog = _Progress(_Spin(), _Text(" Building adata2"), _Bar(), _Time(), transient=True, console=_m6_console)
                _m6_task = _m6_prog.add_task("adata2", total=5)
                _m6_prog.start()
        except Exception:
            _m6_prog = None
    t0 = _step_start("Normalize counts (TF)")
    X1 = adata1.X.tocsr().astype(np.float32) if sp.issparse(adata1.X) else sp.csr_matrix(np.asarray(adata1.X, np.float32))
    rs = np.array(X1.sum(1)).ravel().astype(np.float32); rs[rs == 0] = 1.0
    X1_norm = X1.multiply(1.0 / rs[:, None])
    _step_done("Normalize counts (TF)", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 2) MetaSpot -> meta-domain mapping
    t0 = _step_start("Map metaspots to clusters")
    sub2clu = adata1.obs['meta-domain'].astype(str).values
    clu_codes, clu_uni = pd.factorize(sub2clu, sort=False)
    _step_done("Map metaspots to clusters", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 3) group mean to clusters.  Keep the true aggregate counts separate from
    # the compositional mean used by the downstream enrichment model.  SCVI
    # requires non-negative count-like input; feeding X2 (row-normalized
    # proportions) to a negative-binomial model is statistically invalid.
    t0 = _step_start("Aggregate metaspots -> cluster means")
    ones = np.ones_like(clu_codes, dtype=np.float32)
    G = sp.csr_matrix((ones, (clu_codes, np.arange(len(clu_codes)))), shape=(len(clu_uni), len(clu_codes)))
    row_sum = np.array(G.sum(1)).ravel(); row_sum[row_sum == 0] = 1.0
    X2 = (G.multiply(1.0 / row_sum[:, None]) @ X1_norm).tocsr()
    X2_counts = (G @ X1).tocsr().astype(np.float32, copy=False)
    adata2 = ad.AnnData(X=X2, obs=pd.DataFrame(index=clu_uni), var=adata1.var.copy())
    adata2.layers["counts"] = X2_counts
    adata2.obs["n_metaspots"] = row_sum.astype(np.int64)
    adata2.uns["matrix_semantics"] = {
        "X": "mean MetaSpot transcript-frequency profile; rows sum to approximately one",
        "layers_counts": "sum of raw MetaSpot gene counts per MetaDomain",
    }

    _step_done("Aggregate metaspots -> cluster means", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 4) aggregate geometry stats
    t0 = _step_start("Aggregate geometry stats per cluster")
    geometry_columns = [
        column
        for column in (
            "r_mean",
            "r_std",
            "r_p50",
            "d_cell_norm_mean",
            "d_nuc_norm_mean",
            "elongation",
            "spread",
            "log_n",
        )
        if column in adata1.obs.columns
    ]
    if geometry_columns:
        aggregation = {
            column: (
                "median"
                if column in {"elongation", "spread"}
                else "mean"
            )
            for column in geometry_columns
        }
        agg = (
            adata1.obs[geometry_columns + ["meta-domain"]]
            .groupby("meta-domain", sort=False, observed=True)
            .agg(aggregation)
        )
        adata2.obs = adata2.obs.join(agg, how="left")
    _step_done("Aggregate geometry stats per cluster", t0)
    if _m6_prog and _m6_task is not None:
        try: _m6_prog.advance(_m6_task, 1)
        except Exception: pass
    # 5) representative cell id
    t0 = _step_start("Select representative cell per cluster")
    rep_cell = (adata1.obs[['meta-domain','cell_id']]
              .groupby('meta-domain', observed=True)['cell_id']
              .agg(lambda x: x.value_counts().index[0]))
    adata2.obs['rep_cell_id'] = adata2.obs.index.map(rep_cell.to_dict())
    _step_done("Select representative cell per cluster", t0)
    if _m6_prog and _m6_task is not None:
        try:
            _m6_prog.advance(_m6_task, 1)
            _m6_prog.stop()
        except Exception:
            pass
    return adata2
