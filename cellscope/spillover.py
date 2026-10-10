"""Flag transcripts that probably belong to a neighbouring cell.

Segmentation assigns some transcripts of adjacent cells to the wrong cell. They
form compact patches inside the cell and are easily mistaken for subcellular
structure. A gene that the cell's own type barely expresses is the tell-tale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def relative_type_expression(
    spatial_df: pd.DataFrame, n_types: int = 10, seed: int = 0
) -> np.ndarray:
    """Per transcript: expression of its gene in the cell's type relative to the
    type that expresses the gene most (1 = this type is the top expresser)."""
    cell_codes, cells = pd.factorize(spatial_df["cell_id"])
    gene_codes, genes = pd.factorize(spatial_df["feature_name"].astype(str))
    n_types = min(int(n_types), len(cells))
    if n_types < 2 or len(genes) < 3:
        return np.ones(len(spatial_df))
    counts = sp.csr_matrix(
        (np.ones(len(spatial_df), np.float32), (cell_codes, gene_codes)),
        shape=(len(cells), len(genes)),
    )
    norm = sp.diags(1e4 / np.asarray(counts.sum(1)).ravel()) @ counts
    norm.data = np.log1p(norm.data)
    # shortcut: cell types are a plain k-means on 30 SVD components, upgrade to
    # user-supplied cell types if the sample has rare types k-means would merge.
    n_comp = max(1, min(30, len(genes) - 1, len(cells) - 1))
    embedding = TruncatedSVD(n_comp, random_state=seed).fit_transform(norm)
    cell_type = KMeans(n_types, n_init=10, random_state=seed).fit_predict(embedding)
    membership = sp.csr_matrix(
        (np.ones(len(cells), np.float32), (cell_type, np.arange(len(cells)))),
        shape=(n_types, len(cells)),
    )
    type_counts = (membership @ counts).toarray()
    fraction = type_counts / type_counts.sum(1, keepdims=True).clip(min=1)
    relative = fraction / fraction.max(0, keepdims=True).clip(min=1e-12)
    return relative[cell_type[cell_codes], gene_codes]
