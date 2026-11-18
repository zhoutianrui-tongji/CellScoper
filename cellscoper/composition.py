"""Module 2: local gene composition around each transcript (bag of nearby genes, TF-IDF, SVD), streamed per cell so
no global per-transcript embedding table is materialised."""

from typing import Any, Dict, List, Tuple
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import load_params_yaml
from .runtime import _joblib_backend, _kdtree_query, _preferred_n_jobs, _progress_iter


class _LocalCompositionModel:
    """Compact local-expression transformer used one cell at a time."""

    def __init__(
        self,
        gene2idx: Dict[str, int],
        transformer: Any,
        idf_vec: np.ndarray,
        n_components: int,
        knn_k: int,
        weight_mode: str,
        tf_norm: bool,
        use_tfidf: bool,
    ):
        self.gene2idx = gene2idx
        self.transformer = transformer
        self.idf_vec = np.asarray(idf_vec, dtype=np.float32)
        self.n_components = int(n_components)
        self.knn_k = int(knn_k)
        self.weight_mode = str(weight_mode)
        self.tf_norm = bool(tf_norm)
        self.use_tfidf = bool(use_tfidf)

    @property
    def feature_names(self) -> List[str]:
        return [f"comp_emb{i}" for i in range(self.n_components)]

    def transform_cell(self, dfc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(dfc)
        if n == 0:
            return (
                np.zeros((0, self.n_components), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        points = dfc[["x_location", "y_location"]].to_numpy(np.float32, copy=False)
        ind, dist = _kdtree_query(points, k=min(self.knn_k, n))
        gene_idx = (
            dfc["feature_name"]
            .astype(str)
            .map(self.gene2idx)
            .fillna(-1)
            .to_numpy(dtype=np.int64, copy=False)
        )
        k = ind.shape[1]

        if self.weight_mode == "gaussian":
            radius = dist[:, -1].astype(np.float32) + 1e-6
            weights = np.exp(-((dist / radius[:, None]) ** 2)).astype(np.float32)
        elif self.weight_mode == "inv":
            weights = (1.0 / (dist + 1e-3)).astype(np.float32)
        else:
            weights = np.ones_like(dist, dtype=np.float32)

        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        neighbors = ind.ravel().astype(np.int64, copy=False)
        cols = gene_idx[neighbors]
        values = weights.ravel()
        valid = (cols >= 0) & (cols < len(self.gene2idx))
        bow = sp.coo_matrix(
            (values[valid], (rows[valid], cols[valid])),
            shape=(n, len(self.gene2idx)),
            dtype=np.float32,
        ).tocsr()

        if self.tf_norm:
            row_sum = np.asarray(bow.sum(axis=1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            bow = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        if self.use_tfidf:
            bow = bow.multiply(self.idf_vec[np.newaxis, :]).tocsr()

        z = self.transformer.transform(bow)
        if sp.issparse(z):
            z = z.toarray()
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        if z.shape[1] < self.n_components:
            z = np.pad(z, ((0, 0), (0, self.n_components - z.shape[1])))
        elif z.shape[1] > self.n_components:
            z = z[:, :self.n_components]

        radius = dist[:, -1].astype(np.float32) + 1e-3
        density = (k / (np.pi * radius * radius)).astype(np.float32)
        return z, density

def build_composition_embeddings(point_df: pd.DataFrame, transcripts_df: pd.DataFrame, sampled_cell_ids: List,
                                 show_internal_progress: bool = False) -> Tuple[pd.DataFrame, List[str], "_LocalCompositionModel"]:
    """Module 2: fit the local-composition model; returns (point_df unchanged, feature column names, model).

    Each transcript is described by the distance-weighted bag of the ``top_g`` most frequent genes among its
    ``bow_knn_k`` nearest transcripts of the same cell (term frequency, IDF from a sample of cells), projected to
    ``svd_n_components`` dimensions by a sparse random projection. The model is applied per cell inside Module 3
    (``_LocalCompositionModel.transform_cell``), so no per-transcript embedding table is materialised.
    """
    RANDOM_STATE = 42
    # Build vocabulary
    # Keep only the columns Module 2 actually needs; carrying the full point table
    # through this stage is a major memory multiplier.
    # Avoid huge memory spikes when sampled_cell_ids is a full row-level list.
    use_full_source = False
    try:
        use_full_source = (
            sampled_cell_ids is None
            or len(sampled_cell_ids) >= transcripts_df['cell_id'].nunique(dropna=False)
        )
    except Exception:
        use_full_source = False
    if use_full_source:
        sub_tr_all = transcripts_df
    else:
        sampled_ids = pd.unique(np.asarray(sampled_cell_ids))
        sub_tr_all = transcripts_df[transcripts_df['cell_id'].isin(sampled_ids)]
    gene_counts = sub_tr_all['feature_name'].astype(str).value_counts()
    # defaults; will be overridden by config when available
    TOP_G = 384
    N_COMP = 24
    K_NEI = 32
    MAX_ROWS_PER_CELL_FOR_SVD = 600
    MAX_SVD_ROWS = 120_000
    BOW_WEIGHT_MODE = 'gaussian'  # gaussian | inv | uniform
    USE_TFIDF = True
    TF_NORM = True
    IDF_MODE = 'log1p'  # 'log1p' | 'log' | 'none'
    PARALLEL_CHUNK_CELLS = 128
    try:
        _cfg = load_params_yaml()
        _m2 = _cfg.get('module2', {}) if _cfg else {}
        TOP_G = int(_m2.get('top_g', TOP_G))
        N_COMP = int(_m2.get('svd_n_components', N_COMP))
        K_NEI = int(_m2.get('bow_knn_k', K_NEI))
        MAX_ROWS_PER_CELL_FOR_SVD = int(_m2.get('max_rows_per_cell_for_svd', MAX_ROWS_PER_CELL_FOR_SVD))
        MAX_SVD_ROWS = int(_m2.get('max_svd_rows', MAX_SVD_ROWS))
        BOW_WEIGHT_MODE = str(_m2.get('bow_weight_mode', BOW_WEIGHT_MODE)).lower()
        USE_TFIDF = bool(_m2.get('use_tfidf', USE_TFIDF))
        TF_NORM = bool(_m2.get('tf_norm', TF_NORM))
        IDF_MODE = str(_m2.get('idf_mode', IDF_MODE)).lower()
        PARALLEL_CHUNK_CELLS = max(1, int(_m2.get('parallel_chunk_cells', PARALLEL_CHUNK_CELLS)))
    except Exception:
        pass
    vocab_genes = gene_counts.head(TOP_G).index.astype(str).tolist()
    gene2idx = {g: i for i, g in enumerate(vocab_genes)}

    def _build_knn_per_cell(dfc: pd.DataFrame):
        pts = dfc[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
        if len(pts) == 0:
            return None, None
        ind, dist = _kdtree_query(pts, k=min(K_NEI, len(dfc)))
        return ind, dist

    def _nei_bow_for_indices(
        dfc: pd.DataFrame,
        ind_mat: np.ndarray,
        dist_mat: np.ndarray = None,
        weight_mode: str = 'gaussian',
    ) -> sp.csr_matrix:
        """Vectorized neighborhood BoW with optional distance weights (sparse CSR).
        weight_mode: 'gaussian' | 'inv' | 'uniform'
        """
        G = len(vocab_genes)
        n = len(dfc)
        if n == 0 or G == 0:
            return sp.csr_matrix((n, G), dtype=np.float32)

        gene_idx = (
            dfc['feature_name']
               .astype(str)
               .map(gene2idx).fillna(-1)
               .astype(np.int64, copy=False)
               .to_numpy(copy=False)
        )

        ind = np.asarray(ind_mat)
        if ind.ndim == 1:
            ind = ind[None, :]
        ind = ind.astype(np.int64, copy=False)
        k = ind.shape[1]

        if dist_mat is not None:
            D = np.asarray(dist_mat, dtype=np.float32)
            if D.ndim == 1:
                D = D[None, :]
            D = np.clip(D, 0.0, np.inf)
        else:
            D = None

        # Build weight matrix W aligned with ind's shape
        if D is not None:
            if weight_mode == 'gaussian':
                rk = D[:, min(k - 1, D.shape[1] - 1)].astype(np.float32) + 1e-6
                scale = rk[:, None]
                W = np.exp(- (D / scale) ** 2).astype(np.float32, copy=False)
            elif weight_mode == 'inv':
                W = (1.0 / (D + 1e-3)).astype(np.float32, copy=False)
            else:
                W = np.ones_like(D, dtype=np.float32)
        else:
            W = np.ones_like(ind, dtype=np.float32)

        # Flatten and filter invalid neighbor rows/genes
        row_idx = np.repeat(np.arange(n, dtype=np.int64), k)
        nei_flat = ind.ravel()
        w_flat = W.ravel()

        valid = (nei_flat >= 0) & (nei_flat < n)
        if not valid.any():
            return sp.csr_matrix((n, G), dtype=np.float32)
        nei_flat = nei_flat[valid]
        row_idx = row_idx[valid]
        w_flat = w_flat[valid]

        gi_flat = gene_idx[nei_flat]
        gene_valid = (gi_flat >= 0) & (gi_flat < G)
        if not gene_valid.any():
            return sp.csr_matrix((n, G), dtype=np.float32)

        gi_flat = gi_flat[gene_valid]
        row_idx = row_idx[gene_valid]
        w_flat = w_flat[gene_valid]

        # Build sparse BoW via COO then convert to CSR
        bow = sp.coo_matrix((w_flat.astype(np.float32, copy=False), (row_idx, gi_flat)), shape=(n, G))
        return bow.tocsr()
    # Sample to build SVD training set
    rng = np.random.default_rng(RANDOM_STATE)
    from joblib import Parallel, delayed
    # speed-oriented parallel settings (bounded without env vars)
    # Keep module 2 conservative on memory: use at most 2 workers for large runs.
    N_JOBS = max(1, min(2, _preferred_n_jobs()))
    _backend = _joblib_backend()
    unique_cells = pd.unique(sub_tr_all['cell_id'])
    sample_cell_n = len(unique_cells)
    try:
        _cfg = load_params_yaml() or {}
        _m2 = _cfg.get('module2', {}) if _cfg else {}
        svd_sample_cells = _m2.get('svd_sample_cells')
        if isinstance(svd_sample_cells, (int, float)) and int(svd_sample_cells) > 0:
            sample_cell_n = min(sample_cell_n, int(svd_sample_cells))
    except Exception:
        pass
    # Keep complete cells so the fitted basis sees real spatial neighborhoods.
    # The row budget bounds both the number of cells and retained SVD documents.
    max_cells_by_rows = max(1, MAX_SVD_ROWS // max(1, MAX_ROWS_PER_CELL_FOR_SVD))
    sample_cell_n = max(1, min(int(sample_cell_n), len(unique_cells), max_cells_by_rows))
    sample_cells = rng.choice(unique_cells, size=sample_cell_n, replace=False)

    def _iter_group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    sub_tr_sample = sub_tr_all[sub_tr_all['cell_id'].isin(sample_cells)]
    n_groups_sample = int(sub_tr_sample['cell_id'].nunique(dropna=False)) if len(sub_tr_sample) else 0

    def _svd_tf_block(args):
        cid, dfc = args
        ind, dist = _build_knn_per_cell(dfc)
        if ind is None:
            return None
        bow = _nei_bow_for_indices(dfc, ind, dist, weight_mode=BOW_WEIGHT_MODE)
        if TF_NORM:
            row_sum = np.asarray(bow.sum(1)).ravel().astype(np.float32)
            row_sum[row_sum == 0] = 1.0
            tf = bow.multiply(1.0 / row_sum[:, None]).tocsr()
        else:
            tf = bow.tocsr()
        if tf.shape[0] > MAX_ROWS_PER_CELL_FOR_SVD:
            idx_loc = rng.choice(tf.shape[0], size=MAX_ROWS_PER_CELL_FOR_SVD, replace=False)
            tf = tf[idx_loc]
        return tf

    # Document frequency of every vocabulary gene over the sampled neighbourhoods (for the IDF)
    total_rows = 0
    df_vec = np.zeros(len(vocab_genes), dtype=np.float32)
    sample_chunk_size = PARALLEL_CHUNK_CELLS
    n_sample_chunks = (n_groups_sample + sample_chunk_size - 1) // sample_chunk_size if n_groups_sample else 0
    iterator = _progress_iter(
        _iter_group_chunks(sub_tr_sample, sample_chunk_size),
        desc='Composition IDF sample (parallel)',
        total=n_sample_chunks,
        show=show_internal_progress,
    )
    for chunk in iterator:
        out = Parallel(n_jobs=N_JOBS, backend=_backend, verbose=0)(
            delayed(_svd_tf_block)(pair) for pair in chunk
        )
        for tf in out:
            if tf is None:
                continue
            df_vec += np.asarray(tf.getnnz(axis=0), dtype=np.float32)
            total_rows += tf.shape[0]
            if total_rows >= MAX_SVD_ROWS:
                break
        if total_rows >= MAX_SVD_ROWS:
            break

    from sklearn.random_projection import SparseRandomProjection
    N_rows = float(max(1, total_rows))
    if IDF_MODE == 'none':
        idf_vec = np.ones_like(df_vec, dtype=np.float32)
    elif IDF_MODE == 'log':
        idf_vec = np.log(np.maximum(N_rows / np.maximum(df_vec, 1.0), 1.0)).astype(np.float32)
    else:
        idf_vec = np.log1p(N_rows / (1.0 + df_vec)).astype(np.float32)
    rp_components = max(1, min(N_COMP, len(vocab_genes)))
    rp = SparseRandomProjection(n_components=rp_components, random_state=RANDOM_STATE)
    rp.fit(np.zeros((1, len(vocab_genes)), dtype=np.float32))

    model_idf = (
        idf_vec
        if USE_TFIDF
        else np.ones(len(vocab_genes), dtype=np.float32)
    )
    local_model = _LocalCompositionModel(
        gene2idx=gene2idx,
        transformer=rp,
        idf_vec=model_idf,
        n_components=rp_components,
        knn_k=K_NEI,
        weight_mode=BOW_WEIGHT_MODE,
        tf_norm=TF_NORM,
        use_tfidf=USE_TFIDF,
    )
    geom_cols = ['r_signed', 'd_cell_norm', 'd_nuc_norm'] + (
        ['sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta'] if 'sin_theta' in point_df.columns else []
    )
    gmm_cols = geom_cols + local_model.feature_names + ['local_density']
    gc.collect()
    return point_df, gmm_cols, local_model
