"""Modules 3-4: MetaSpots (spatially connected pieces of each cell side) and the MetaSpot AnnData (adata1)."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
import gc
import logging as _logging
import os
from sklearn.cluster import AgglomerativeClustering
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .composition import _LocalCompositionModel
from .keys import METASPOT_COL, METASPOT_LOCAL_COL
from .runtime import (
    _quiet_expected_disconnected_connectivity_warning,
    _cluster_chunk_size,
    _coerce_bool_series,
    _joblib_backend,
    _kdtree_query,
    _preferred_n_jobs,
    _progress_iter,
)


def _safe_standardize(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    mu = X.mean(0, dtype=np.float64)
    sd = X.std(0, dtype=np.float64)
    good = sd > 0
    if not np.any(good):
        return np.zeros_like(X, np.float32), good
    Xg = X[:, good].astype(np.float32, copy=False)
    Xz = (Xg - mu[good].astype(np.float32)) / sd[good].astype(np.float32)
    return Xz, good

def _select_top_variance_features(X: np.ndarray, n_features: int) -> np.ndarray:
    """Select per-cell features without creating a dataset-wide dense matrix."""
    X = np.asarray(X, dtype=np.float32)
    limit = int(n_features)
    if limit <= 0 or X.ndim != 2 or X.shape[1] <= limit:
        return X
    finite = np.where(np.isfinite(X), X, 0.0)
    variances = np.var(finite, axis=0, dtype=np.float64)
    order = np.argsort(-variances, kind="stable")[:limit]
    return X[:, order]

def _target_metaspot_counts_from_site_counts(
    site_counts: np.ndarray,
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> np.ndarray:
    """Vectorize the configured MetaSpot budget over per-cell site counts."""
    n_sites = np.asarray(site_counts, dtype=np.int64)
    if np.any(n_sites < 0):
        raise ValueError("site_counts must be non-negative")

    target_sites = max(1, int(target_sites_per_metaspot))
    target_count = (n_sites + target_sites - 1) // target_sites
    lower_bound = max(1, int(min_metaspots_per_cell))
    target_count = np.maximum(lower_bound, target_count)
    if isinstance(max_metaspots_per_cell, (int, float)):
        upper_bound = max(lower_bound, int(max_metaspots_per_cell))
        target_count = np.minimum(target_count, upper_bound)
    target_count = np.minimum(target_count, n_sites)
    return np.maximum(1, target_count).astype(np.int64, copy=False)

def _target_metaspot_count(
    coordinates: np.ndarray,
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> int:
    """Choose a per-cell MetaSpot count from the number of unique XY sites."""
    xy = np.asarray(coordinates, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_points, 2)")
    finite_xy = xy[np.isfinite(xy).all(axis=1)]
    n_sites = int(np.unique(finite_xy, axis=0).shape[0])
    return int(
        _target_metaspot_counts_from_site_counts(
            np.asarray([n_sites], dtype=np.int64),
            target_sites_per_metaspot=target_sites_per_metaspot,
            min_metaspots_per_cell=min_metaspots_per_cell,
            max_metaspots_per_cell=max_metaspots_per_cell,
        )[0]
    )

def _allocate_partition_target_counts(
    partition_coordinates: Sequence[np.ndarray],
    target_sites_per_metaspot: int,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> List[int]:
    """Allocate one cell-level MetaSpot budget across spatial compartments.

    Every non-empty compartment receives at least one MetaSpot.  The remaining
    budget is assigned in proportion to unique XY sites, with deterministic
    largest-remainder tie breaking.  This preserves the configured per-cell
    cap while preventing any MetaSpot from crossing a compartment boundary.
    """
    partitions = [
        np.asarray(coordinates, dtype=np.float32)
        for coordinates in partition_coordinates
        if len(coordinates)
    ]
    if not partitions:
        return []

    site_counts = np.asarray(
        [
            np.unique(
                coordinates[np.isfinite(coordinates).all(axis=1)],
                axis=0,
            ).shape[0]
            for coordinates in partitions
        ],
        dtype=np.int64,
    )
    site_counts = np.maximum(site_counts, 1)
    combined = np.vstack(partitions)
    total_target = _target_metaspot_count(
        combined,
        target_sites_per_metaspot=target_sites_per_metaspot,
        min_metaspots_per_cell=max(
            int(min_metaspots_per_cell),
            len(partitions),
        ),
        max_metaspots_per_cell=max_metaspots_per_cell,
    )
    total_target = max(len(partitions), int(total_target))
    total_target = min(int(site_counts.sum()), total_target)

    raw = total_target * site_counts.astype(np.float64) / site_counts.sum()
    allocation = np.floor(raw).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, site_counts)

    while int(allocation.sum()) < total_target:
        candidates = np.flatnonzero(allocation < site_counts)
        if candidates.size == 0:
            break
        residual = raw[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmax(residual)])
        allocation[chosen] += 1

    while int(allocation.sum()) > total_target:
        candidates = np.flatnonzero(allocation > 1)
        if candidates.size == 0:
            break
        residual = raw[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmin(residual)])
        allocation[chosen] -= 1

    return allocation.astype(int).tolist()

def _enforce_spatially_connected_labels(
    coordinates: np.ndarray,
    labels: np.ndarray,
    feature_matrix: Optional[np.ndarray] = None,
    target_n_clusters: Optional[int] = None,
    spatial_compactness: float = 1.0,
    connectivity_k: int = 8,
    radius_quantile: float = 0.9,
    radius_factor: float = 1.5,
    smoothing_iterations: int = 1,
    smoothing_min_fraction: float = 0.6,
    min_component_size: int = 20,
    spatial_share: Optional[float] = None,
) -> np.ndarray:
    """Turn feature-space labels into spatially connected 2D components.

    ``spatial_share`` (0-1), when given, replaces ``spatial_compactness``: the coordinate block is scaled so that it
    carries this share of the total feature variance (isotropic in x, y, z), whatever the number of feature columns.

    Duplicate transcript coordinates are collapsed to one spatial site, labels
    are re-clustered with an XY-constrained Ward model when features are given,
    and undersized components are merged across a shared boundary. Without a
    feature matrix, the preliminary labels are smoothed and split directly.
    Component sizes are measured in transcripts, not unique XY sites.
    """
    xy = np.asarray(coordinates, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)
    n_points = len(point_labels)
    if n_points == 0:
        return point_labels.copy()
    if xy.ndim != 2 or xy.shape[0] != n_points or xy.shape[1] not in (2, 3):
        raise ValueError("coordinates must have shape (n_points, 2) or (n_points, 3)")
    features = None
    if feature_matrix is not None:
        features = np.asarray(feature_matrix, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != n_points:
            raise ValueError("feature_matrix must have n_points rows")

    finite = np.isfinite(xy).all(axis=1)
    if not finite.all():
        # Module 3 already drops invalid XY rows; keep this helper defensive.
        out = point_labels.copy()
        if finite.any():
            out[finite] = _enforce_spatially_connected_labels(
                xy[finite],
                point_labels[finite],
                feature_matrix=features[finite] if features is not None else None,
                target_n_clusters=target_n_clusters,
                spatial_compactness=spatial_compactness,
                connectivity_k=connectivity_k,
                radius_quantile=radius_quantile,
                radius_factor=radius_factor,
                smoothing_iterations=smoothing_iterations,
                smoothing_min_fraction=smoothing_min_fraction,
                min_component_size=min_component_size,
                spatial_share=spatial_share,
            )
        return out

    unique_xy, inverse, site_weights = np.unique(
        xy,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    n_sites = len(unique_xy)
    if n_sites <= 1:
        return np.zeros(n_points, dtype=np.int32)

    # Canonical spatial Ward segmentation starts from one constant label. Skip
    # thousands of tiny majority-vote reductions in that common path.
    point_label_values = np.unique(point_labels)
    if point_label_values.size == 1:
        site_labels = np.full(
            n_sites,
            int(point_label_values[0]),
            dtype=np.int32,
        )
    else:
        # Pick a deterministic majority label at coordinates containing
        # multiple transcripts.
        site_labels = np.empty(n_sites, dtype=np.int32)
        order = np.argsort(inverse, kind="stable")
        sorted_sites = inverse[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_sites)) + 1]
        stops = np.r_[starts[1:], len(order)]
        for start, stop in zip(starts, stops):
            site = int(sorted_sites[start])
            values, counts = np.unique(
                point_labels[order[start:stop]],
                return_counts=True,
            )
            site_labels[site] = int(values[np.argmax(counts)])

    query_k = max(2, min(n_sites, int(connectivity_k) + 1))
    neighbor_idx, neighbor_dist = _kdtree_query(unique_xy, query_k)
    if neighbor_idx is None or neighbor_dist is None:
        return point_labels.copy()

    positive_dist = np.where(neighbor_dist > 1e-8, neighbor_dist, np.inf)
    nearest = positive_dist.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return point_labels.copy()

    q = float(np.clip(radius_quantile, 0.0, 1.0))
    base_spacing = float(np.quantile(nearest, q))
    radius = base_spacing * max(float(radius_factor), 1e-6)
    if not np.isfinite(radius) or radius <= 0:
        return point_labels.copy()

    edge_rows = np.repeat(
        np.arange(n_sites, dtype=np.int64),
        neighbor_idx.shape[1],
    )
    edge_cols = np.asarray(neighbor_idx, dtype=np.int64).ravel()
    edge_distances = np.asarray(neighbor_dist, dtype=np.float64).ravel()
    valid_edges = (
        (edge_rows != edge_cols)
        & np.isfinite(edge_distances)
        & (edge_distances > 0.0)
    )
    edge_rows = edge_rows[valid_edges]
    edge_cols = edge_cols[valid_edges]
    edge_distances = edge_distances[valid_edges]
    edge_low = np.minimum(edge_rows, edge_cols)
    edge_high = np.maximum(edge_rows, edge_cols)
    edge_codes = edge_low * np.int64(n_sites) + edge_high
    knn_codes = np.unique(edge_codes)
    local_codes = np.unique(edge_codes[edge_distances <= radius])

    # Feature-aware Ward clustering needs one connected neighborhood graph to
    # preserve its requested cluster count. Direct label splitting keeps the
    # stricter native-spacing radius so genuinely separated islands stay split.
    selected_codes = knn_codes if features is not None else local_codes
    if selected_codes.size == 0:
        return point_labels.copy()

    edge_a = (selected_codes // np.int64(n_sites)).astype(
        np.int32,
        copy=False,
    )
    edge_b = (selected_codes % np.int64(n_sites)).astype(
        np.int32,
        copy=False,
    )

    used_constrained_clustering = False
    if features is not None and n_sites > 1:
        n_features = features.shape[1]
        site_features = np.zeros((n_sites, n_features), dtype=np.float32)
        np.add.at(site_features, inverse, features)
        site_features /= site_weights[:, None].astype(np.float32)
        site_features, _ = _safe_standardize(site_features)
        if unique_xy.shape[1] == 3:
            # One length scale for all axes (the x-y spread): standardising z on its
            # own would stretch a ~4 um cell depth to the ~20 um x-y extent.
            centred = unique_xy - unique_xy.mean(axis=0)
            scale = float(np.sqrt(np.mean(np.var(unique_xy[:, :2], axis=0))))
            xy_features = (centred / max(scale, 1e-6)).astype(np.float32)
        else:
            xy_features, _ = _safe_standardize(unique_xy)
        weight = max(float(spatial_compactness), 0.0)
        if spatial_share is not None:
            # 坐标块方差 = share / (1 - share) × 特征块方差；特征列已逐列标准化，方差和 = 非常数列数
            share = float(np.clip(spatial_share, 0.0, 0.95))
            feature_var = float(np.var(site_features, axis=0).sum())
            xy_var = float(np.var(xy_features, axis=0).sum())
            weight = float(np.sqrt(share / (1.0 - share) * max(feature_var, 1.0) / max(xy_var, 1e-12)))
        combined_features = np.hstack(
            [
                site_features,
                xy_features * weight,
            ]
        ).astype(np.float32, copy=False)

        requested_clusters = (
            int(target_n_clusters)
            if isinstance(target_n_clusters, (int, float))
            else int(np.unique(point_labels).size)
        )
        requested_clusters = max(1, min(requested_clusters, n_sites))
        if requested_clusters > 1:
            rows = np.concatenate((edge_a, edge_b))
            cols = np.concatenate((edge_b, edge_a))
            connectivity = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_sites, n_sites),
            )
            try:
                # Some runners prepend their own warning filters after this
                # module is imported, so reassert the precise rule at the call
                # site without wrapping or serializing parallel clustering.
                _quiet_expected_disconnected_connectivity_warning()
                site_labels = AgglomerativeClustering(
                    n_clusters=requested_clusters,
                    linkage="ward",
                    connectivity=connectivity,
                    compute_full_tree=False,
                ).fit_predict(combined_features).astype(np.int32)
                used_constrained_clustering = True
            except Exception as exc:
                _logging.getLogger("cellscoper").warning(
                    "Spatially constrained clustering failed; using label "
                    "smoothing fallback: %s",
                    exc,
                )

    if not used_constrained_clustering:
        adjacency = [[] for _ in range(n_sites)]
        for a, b in zip(edge_a, edge_b):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        smooth_fraction = float(np.clip(smoothing_min_fraction, 0.5, 1.0))
        for _ in range(max(0, int(smoothing_iterations))):
            updated = site_labels.copy()
            changed = False
            for i, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                votes: Dict[int, int] = defaultdict(int)
                votes[int(site_labels[i])] += int(site_weights[i])
                for j in neighbors:
                    votes[int(site_labels[j])] += int(site_weights[j])
                current = int(site_labels[i])
                winner = max(
                    votes,
                    key=lambda label: (votes[label], label == current, -int(label)),
                )
                total = sum(votes.values())
                if (
                    winner != current
                    and total > 0
                    and votes[winner] / total >= smooth_fraction
                ):
                    updated[i] = int(winner)
                    changed = True
            site_labels = updated
            if not changed:
                break

    # First split every original label into its XY-connected components. A
    # component's root is its smallest site index (union keeps the smaller root).
    same = site_labels[edge_a] == site_labels[edge_b]
    from scipy.sparse.csgraph import connected_components

    _, component = connected_components(
        sp.csr_matrix(
            (np.ones(int(same.sum()), dtype=np.int8), (edge_a[same], edge_b[same])),
            shape=(n_sites, n_sites),
        ),
        directed=False,
    )
    smallest = np.full(component.max() + 1, n_sites, dtype=np.int64)
    np.minimum.at(smallest, component, np.arange(n_sites))
    parent = smallest[component].astype(np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return int(node)

    def union(a: int, b: int) -> bool:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return False
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        return True

    def all_roots() -> np.ndarray:
        roots = parent.copy()
        while True:
            jumped = roots[roots]
            if np.array_equal(jumped, roots):
                return roots
            roots = jumped

    min_size = max(1, int(min_component_size))
    for _ in range(32):
        roots = all_roots()
        parent[:] = roots
        sizes = np.bincount(roots, weights=site_weights, minlength=n_sites)
        present = np.unique(roots)
        component_size: Dict[int, int] = {
            int(root): int(sizes[root]) for root in present
        }
        small_roots = {int(root) for root in present if sizes[root] < min_size}
        if not small_roots:
            break

        # Contacts between components, each neighbour listed in the order the
        # edges first meet it (the original Counter insertion order; it breaks ties).
        root_a, root_b = roots[edge_a], roots[edge_b]
        across = np.flatnonzero(root_a != root_b)
        source = np.empty(2 * across.size, dtype=np.int64)
        target = np.empty(2 * across.size, dtype=np.int64)
        source[0::2], target[0::2] = root_a[across], root_b[across]
        source[1::2], target[1::2] = root_b[across], root_a[across]
        pair_codes = source * np.int64(n_sites) + target
        unique_pairs, first_seen, pair_counts = np.unique(
            pair_codes, return_index=True, return_counts=True
        )
        pair_source = unique_pairs // np.int64(n_sites)
        keep = np.isin(pair_source, list(small_roots))
        order = np.argsort(first_seen[keep], kind="stable")
        boundary_contacts: Dict[int, Dict[int, int]] = defaultdict(dict)
        for src, dst, count in zip(
            pair_source[keep][order],
            (unique_pairs % np.int64(n_sites))[keep][order],
            pair_counts[keep][order],
        ):
            boundary_contacts[int(src)][int(dst)] = int(count)

        changed = False
        for root in sorted(small_roots, key=lambda r: (component_size[r], r)):
            root = find(root)
            if component_size.get(root, 0) >= min_size:
                continue
            candidates = {
                find(candidate): contacts
                for candidate, contacts in boundary_contacts.get(root, {}).items()
                if find(candidate) != root
            }
            if not candidates:
                continue
            target_root = max(
                candidates,
                key=lambda candidate: (
                    candidates[candidate],
                    component_size.get(candidate, 0),
                    -int(candidate),
                ),
            )
            changed |= union(root, target_root)
        if not changed:
            break

    roots = all_roots()
    # Number components in order of their first site.
    _, first_site, root_index = np.unique(roots, return_index=True, return_inverse=True)
    connected_site_labels = np.argsort(np.argsort(first_site, kind="stable"), kind="stable")[
        root_index
    ].astype(np.int32)
    return connected_site_labels[inverse]

def per_cell_clustering(
    point_df: pd.DataFrame,
    gmm_cols: List[str],
    composition_model: Optional[_LocalCompositionModel] = None,
    show_internal_progress: bool = False,
    min_segment_points: int = 30,
    n_segmentation_features: int = 24,
    small_backend_cells: int = 12000,
    sequential_cells: int = 256,
    spatial_connectivity_k: int = 8,
    spatial_radius_quantile: float = 0.9,
    spatial_radius_factor: float = 1.5,
    spatial_smoothing_iterations: int = 1,
    spatial_smoothing_min_fraction: float = 0.6,
    spatial_min_component_size: Optional[int] = None,
    spatial_compactness: float = 1.0,
    target_sites_per_metaspot: int = 15,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
    parallel_chunk_cells: Optional[int] = None,
    parallel_n_jobs: Optional[int] = None,
    use_z_location: bool = False,
    segmentation_geometry: str = "legacy",
    segmentation_spatial_share: float = 1.0 / 3.0,
    ) -> pd.DataFrame:
    """Module 3: split every cell side (nucleus / cytoplasm) into spatially connected MetaSpots.

    Spatially constrained Ward clustering of each cell side's transcripts on standardised geometry + local composition
    features (the ``n_segmentation_features`` with the largest variance), with about one MetaSpot per
    ``target_sites_per_metaspot`` distinct x-y sites (bounded by ``min/max_metaspots_per_cell``); in 3D when
    ``use_z_location`` and z is present. Cell sides with fewer than ``min_segment_points`` transcripts form one MetaSpot.
    """
    MIN_SUB_PTS = int(min_segment_points)
    if not isinstance(target_sites_per_metaspot, (int, float)) or int(target_sites_per_metaspot) < 1:
        raise ValueError("module3.target_sites_per_metaspot must be a positive integer")
    TARGET_SITES_PER_METASPOT = int(target_sites_per_metaspot)
    SPATIAL_MIN_COMPONENT_SIZE = (
        int(spatial_min_component_size)
        if isinstance(spatial_min_component_size, (int, float))
        else MIN_SUB_PTS
    )

    def _finalize_spatial_labels(
        g: pd.DataFrame,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
        target_count_override: Optional[int] = None,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32)
        if len(labels) <= 1:
            return labels
        coordinates = g[["x_location", "y_location"]].to_numpy(
            np.float32,
            copy=False,
        )
        if isinstance(target_count_override, (int, float)):
            target_count = max(1, int(target_count_override))
        else:
            target_count = _target_metaspot_count(  # MetaSpot count from x-y sites either way
                coordinates,
                target_sites_per_metaspot=TARGET_SITES_PER_METASPOT,
                min_metaspots_per_cell=min_metaspots_per_cell,
                max_metaspots_per_cell=max_metaspots_per_cell,
            )
        if use_z_location and "z_location" in g.columns and np.isfinite(g["z_location"].to_numpy()).all():
            # MetaSpots split by height too: in x-y only they spanned ~2.7 of a ~3.7 um
            # cell depth (Atera BRCA), mixing small structures with RNA above/below.
            coordinates = g[["x_location", "y_location", "z_location"]].to_numpy(np.float32, copy=False)
        return _enforce_spatially_connected_labels(
            coordinates,
            labels,
            feature_matrix=features,
            target_n_clusters=target_count,
            spatial_compactness=spatial_compactness,
            spatial_share=float(segmentation_spatial_share) if isotropic else None,
            connectivity_k=spatial_connectivity_k,
            radius_quantile=spatial_radius_quantile,
            radius_factor=spatial_radius_factor,
            smoothing_iterations=spatial_smoothing_iterations,
            smoothing_min_fraction=spatial_smoothing_min_fraction,
            min_component_size=SPATIAL_MIN_COMPONENT_SIZE,
        )

    # Select features with the highest variance. If per-row composition columns
    # are not materialized in `point_df` (to avoid large memory), a
    # precomputed per-cell composition DataFrame may be present in the
    # environment and will be used when building per-transcript features.
    # We build a modest sample by combining per-row geometry and per-cell
    # composition (looked up) to compute variances without expanding global
    # comp arrays.
    geom_cols = [c for c in gmm_cols if c in ('x_location', 'y_location', 'r_signed', 'd_cell_norm', 'd_nuc_norm', 'sin_theta', 'cos_theta', 'sin2_theta', 'cos2_theta', 'local_density')]
    comp_cols = [c for c in gmm_cols if c not in geom_cols]
    # isotropic：分割只用局部基因组成 + 局部密度；空间紧致性只来自 x-y(-z) 坐标，各方向等权，
    # 坐标所占方差份额 = segmentation_spatial_share。legacy 另含径向距离（r_signed、d_nuc_norm、d_cell_norm，
    # 胞质中三者相关 |r| ≥ 0.995）与绕核夹角（sin/cos θ、2θ），使 MetaSpot 沿绕核方向拉长。
    if segmentation_geometry not in ("legacy", "isotropic"):
        raise ValueError("module3.segmentation_geometry must be 'legacy' or 'isotropic'")
    isotropic = segmentation_geometry == "isotropic"
    if isotropic:
        geom_cols = [c for c in geom_cols if c == 'local_density']

    # Per-cell MetaSpot segmentation (parallel batching)

    def _iter_group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    n_groups = int(point_df['cell_id'].nunique(dropna=False)) if len(point_df) else 0

    def _fit_one(
        args,
        allow_nucleus_partition: bool = True,
        target_count_override: Optional[int] = None,
    ):
        cid, g = args
        # Ensure minimal geometry present; composition may be provided per-cell
        g = g.dropna(subset=['x_location', 'y_location'])
        if len(g) == 0:
            return cid, pd.Series([], dtype=np.int32, index=g.index)

        # MetaSpots never cross the nuclear boundary: each side is segmented alone.
        if allow_nucleus_partition:
            if "in_nucleus" in g.columns:
                nucleus_flags = _coerce_bool_series(
                    g["in_nucleus"],
                    default=False,
                )
            elif "r_signed" in g.columns:
                nucleus_flags = (
                    pd.to_numeric(g["r_signed"], errors="coerce")
                    .fillna(0.0)
                    .lt(0.0)
                )
            else:
                raise ValueError(
                    "Module 3 needs either 'in_nucleus' or 'r_signed' in the point table"
                )

            present_partitions = [
                flag
                for flag in (True, False)
                if bool(nucleus_flags.eq(flag).any())
            ]
            if len(present_partitions) > 1:
                partition_frames = [
                    g.loc[nucleus_flags.eq(flag)]
                    for flag in present_partitions
                ]
                partition_targets = _allocate_partition_target_counts(
                    [
                        frame[
                            ["x_location", "y_location"]
                        ].to_numpy(np.float32, copy=False)
                        for frame in partition_frames
                    ],
                    target_sites_per_metaspot=TARGET_SITES_PER_METASPOT,
                    min_metaspots_per_cell=min_metaspots_per_cell,
                    max_metaspots_per_cell=max_metaspots_per_cell,
                )

                combined = pd.Series(
                    index=g.index,
                    dtype=np.int32,
                )
                label_offset = 0
                for frame, partition_target in zip(
                    partition_frames,
                    partition_targets,
                ):
                    _, partition_labels = _fit_one(
                        (cid, frame),
                        allow_nucleus_partition=False,
                        target_count_override=partition_target,
                    )
                    dense_labels = pd.factorize(
                        partition_labels,
                        sort=True,
                    )[0].astype(np.int32)
                    combined.loc[partition_labels.index] = (
                        dense_labels + label_offset
                    )
                    label_offset += max(
                        1,
                        int(np.unique(dense_labels).size),
                    )
                return cid, combined.astype(np.int32)

        n = len(g)
        if n < MIN_SUB_PTS:
            labels = np.zeros(n, dtype=np.int32)
            labels = _finalize_spatial_labels(
                g,
                labels,
                target_count_override=target_count_override,
            )
            return cid, pd.Series(labels, index=g.index, dtype=np.int32)
        # Build local expression features for this cell only, then release them
        # after clustering. This avoids materializing dense embeddings globally.
        local_density = None
        if composition_model is not None:
            X_comp, local_density = composition_model.transform_cell(g)
        else:  # no Module 2 model: composition columns already in the table, missing ones as zeros
            X_comp = g[[c for c in comp_cols if c in g.columns]].to_numpy(np.float32, copy=False)
            if X_comp.shape[1] < len(comp_cols):
                missing = len(comp_cols) - X_comp.shape[1]
                X_comp = np.hstack([X_comp, np.zeros((len(g), missing), dtype=np.float32)])
        # geometry
        geom_use = [c for c in geom_cols if c in g.columns and c != 'local_density']
        if len(geom_use) == 0:
            X_geom = np.zeros((len(g), 2), dtype=np.float32)
        else:
            X_geom = g[geom_use].to_numpy(np.float32, copy=False)
        if local_density is not None:
            X_geom = np.column_stack((X_geom, local_density)).astype(np.float32, copy=False)
        X_full = np.hstack([X_geom, X_comp]) if X_comp.size else X_geom
        X_full = _select_top_variance_features(X_full, n_segmentation_features)
        Xz, good = _safe_standardize(X_full)
        if Xz.shape[1] == 0 or not np.any(good):
            X_coord = g[['x_location', 'y_location']].to_numpy(np.float32, copy=False)
            Xz, _ = _safe_standardize(X_coord)
        # Spatial resolution determines the number of MetaSpots (spatially constrained Ward).
        labels = _finalize_spatial_labels(
            g,
            np.zeros(n, dtype=np.int32),
            Xz,
            target_count_override=target_count_override,
        )
        return cid, pd.Series(labels, index=g.index, dtype=np.int32)

    # Lightweight backend selection for small datasets to cut scheduler/IPC overhead
    _force_threads = bool(os.environ.get("CELLSCOPER_FORCE_THREADS", "").strip())
    use_sequential = n_groups <= int(sequential_cells)
    _backend = _joblib_backend()
    if _force_threads or n_groups <= int(small_backend_cells):
        _backend = "threading"
    if use_sequential:
        iterator = _progress_iter(range(n_groups), desc='MetaSpot segmentation (seq)', total=n_groups, show=show_internal_progress)
        labels_per_cell = pd.Series(
            -1,
            index=point_df.index,
            dtype=np.int32,
        )
        for chunk in _iter_group_chunks(point_df, 1):
            for pair in chunk:
                _, labels = _fit_one(pair)
                labels_per_cell.loc[labels.index] = labels.to_numpy(
                    dtype=np.int32,
                    copy=False,
                )
        point_df = point_df.copy(deep=False)
        point_df[METASPOT_LOCAL_COL] = labels_per_cell
        missing = point_df[METASPOT_LOCAL_COL].isna()
        if missing.any():
            point_df.loc[missing, METASPOT_LOCAL_COL] = -1
        point_df[METASPOT_LOCAL_COL] = point_df[METASPOT_LOCAL_COL].astype(np.int32)
        point_df[METASPOT_COL] = (
            point_df['cell_id'].astype(str)
            + ':'
            + point_df[METASPOT_LOCAL_COL].astype(str)
        )
        return point_df

    from joblib import Parallel, delayed
    # Ward/k-d tree work releases the GIL and each cell is bounded in size.
    # Use several cells concurrently by default while keeping an explicit
    # upper bound for large production runs.
    available_jobs = _preferred_n_jobs()
    if isinstance(parallel_n_jobs, (int, float)) and int(parallel_n_jobs) > 0:
        N_JOBS = max(1, min(int(parallel_n_jobs), available_jobs))
    else:
        N_JOBS = max(1, min(8, available_jobs))
    # _backend already chosen above, but keep env override precedence
    if os.environ.get("CELLSCOPER_FORCE_THREADS", "").strip():
        _backend = "threading"

    # prefer larger chunk to reduce scheduler overhead
    chunk_size = max(64, _cluster_chunk_size(parallel_chunk_cells))
    labels_per_cell = pd.Series(
        -1,
        index=point_df.index,
        dtype=np.int32,
    )
    n_chunks = (n_groups + chunk_size - 1) // chunk_size if n_groups else 0
    iterator = _progress_iter(
        _iter_group_chunks(point_df, chunk_size),
        desc='MetaSpot segmentation (parallel)',
        total=n_chunks,
        show=show_internal_progress,
    )
    _logging.getLogger("cellscoper").info(
        "Module 3 parallel segmentation | cells=%d | workers=%d | backend=%s | chunk_cells=%d",
        n_groups,
        N_JOBS,
        _backend,
        chunk_size,
    )
    # Keep one executor for all chunks.  Recreating it for every small chunk
    # dominated scheduling time on large COAD runs.
    from contextlib import nullcontext

    thread_limit = nullcontext()
    if N_JOBS > 1:
        try:
            from threadpoolctl import threadpool_limits

            # Prevent every concurrently segmented cell from starting another
            # BLAS/OpenMP pool. This context is process-wide and therefore
            # belongs outside the persistent joblib executor.
            thread_limit = threadpool_limits(limits=1)
        except Exception:
            thread_limit = nullcontext()
    with thread_limit:
        with Parallel(
            n_jobs=N_JOBS,
            backend=_backend,
            verbose=0,
            pre_dispatch=f"3*{N_JOBS}",
            batch_size="auto",
            max_nbytes=None,
        ) as parallel:
            for chunk_index, chunk in enumerate(iterator, start=1):
                try:
                    out = parallel(delayed(_fit_one)(pair) for pair in chunk)
                except Exception as e:
                    # Fallback: sequential processing to avoid hard failure
                    # (for example, a worker memory allocation failure).
                    print(
                        "[WARN] Parallel cluster chunk failed, fallback "
                        f"sequential: {e}"
                    )
                    out = [_fit_one(pair) for pair in chunk]
                if out:
                    chunk_labels = pd.concat(
                        [series for _, series in out]
                    ).astype(np.int32, copy=False)
                    labels_per_cell.loc[
                        chunk_labels.index
                    ] = chunk_labels.to_numpy(
                        dtype=np.int32,
                        copy=False,
                    )
                del out
                # Periodic collection retains bounded temporary lifetimes
                # without making collection the primary segmentation cost.
                if chunk_index % 16 == 0:
                    gc.collect()
    gc.collect()

    point_df = point_df.copy(deep=False)
    point_df[METASPOT_LOCAL_COL] = labels_per_cell
    missing = point_df[METASPOT_LOCAL_COL].isna()
    if missing.any():
        point_df.loc[missing, METASPOT_LOCAL_COL] = -1
    point_df[METASPOT_LOCAL_COL] = point_df[METASPOT_LOCAL_COL].astype(np.int32)
    point_df[METASPOT_COL] = (
        point_df['cell_id'].astype(str)
        + ':'
        + point_df[METASPOT_LOCAL_COL].astype(str)
    )
    return point_df

def _metaspot_segmentation_audit(
    point_df: pd.DataFrame,
    *,
    target_sites_per_metaspot: Optional[int] = None,
    min_metaspots_per_cell: int = 1,
    max_metaspots_per_cell: Optional[int] = None,
) -> pd.DataFrame:
    """Return compact per-cell checks for MetaSpot size and compartment mixing."""
    required = {
        "cell_id",
        METASPOT_COL,
        "x_location",
        "y_location",
    }
    missing = sorted(required.difference(point_df.columns))
    if missing:
        raise KeyError(
            "MetaSpot segmentation audit requires columns: "
            + ", ".join(missing)
        )

    if "in_nucleus" in point_df.columns:
        in_nucleus = _coerce_bool_series(point_df["in_nucleus"], default=False).to_numpy(dtype=bool)
    elif "r_signed" in point_df.columns:
        in_nucleus = pd.to_numeric(point_df["r_signed"], errors="coerce").fillna(0.0).lt(0.0).to_numpy()
    else:
        in_nucleus = np.zeros(len(point_df), dtype=bool)
    x = point_df["x_location"].to_numpy()
    y = point_df["y_location"].to_numpy()
    # 全部用整数编号（按首次出现顺序，与 groupby(sort=False) 一致）计数：字符串键与 drop_duplicates 复制整表在
    # 几亿行时是主要耗时。缺失的 cell/MetaSpot 编号为 -1，与原先一样作为一组。
    cell_codes, cell_labels = pd.factorize(point_df["cell_id"])
    spot_codes = pd.factorize(point_df[METASPOT_COL])[0]
    width = np.int64(spot_codes.max() + 2) if len(spot_codes) else np.int64(1)
    pair, pair_keys = pd.factorize((cell_codes.astype(np.int64) + 1) * width + (spot_codes.astype(np.int64) + 1))
    n_pairs = len(pair_keys)
    n_tx = np.bincount(pair, minlength=n_pairs)
    n_inside = np.bincount(pair, weights=in_nucleus, minlength=n_pairs)
    first_site = ~pd.DataFrame({"p": pair, "x": x, "y": y}).duplicated().to_numpy()
    spot_stats = pd.DataFrame(
        {
            "cell_id": np.asarray(pair_keys) // width - 1,
            "n_transcripts": n_tx,
            "nucleus_min": n_inside == n_tx,
            "nucleus_max": n_inside > 0,
            "n_unique_sites": np.bincount(pair[first_site], minlength=n_pairs),
        }
    )
    spot_stats["nucleus_mixed"] = (
        spot_stats["nucleus_min"] != spot_stats["nucleus_max"]
    )
    spot_stats["mixed_transcripts"] = (
        spot_stats["n_transcripts"]
        * spot_stats["nucleus_mixed"].astype(np.int64)
    )

    per_cell = spot_stats.groupby(
        "cell_id",
        sort=False,
        observed=True,
    ).agg(
        actual_metaspots=("n_transcripts", "size"),
        median_transcripts_per_metaspot=("n_transcripts", "median"),
        max_transcripts_per_metaspot=("n_transcripts", "max"),
        median_sites_per_metaspot=("n_unique_sites", "median"),
        max_sites_per_metaspot=("n_unique_sites", "max"),
        nucleus_mixed_metaspots=("nucleus_mixed", "sum"),
        mixed_transcripts=("mixed_transcripts", "sum"),
        n_transcripts=("n_transcripts", "sum"),
    )
    per_cell["nucleus_mixed_metaspot_fraction"] = (
        per_cell["nucleus_mixed_metaspots"]
        / per_cell["actual_metaspots"].clip(lower=1)
    )
    per_cell["nucleus_mixed_transcript_fraction"] = (
        per_cell["mixed_transcripts"]
        / per_cell["n_transcripts"].clip(lower=1)
    )

    first_cell_site = ~pd.DataFrame({"c": cell_codes, "x": x, "y": y}).duplicated().to_numpy()
    site_cells = cell_codes[first_cell_site]
    site_counts = pd.Series(site_cells).value_counts()
    per_cell["n_unique_sites"] = site_counts.reindex(per_cell.index).astype(int)
    if isinstance(target_sites_per_metaspot, (int, float)):
        unique_xy = np.column_stack([x[first_cell_site], y[first_cell_site]]).astype(np.float32, copy=False)
        finite_site_mask = np.isfinite(unique_xy).all(axis=1)
        finite_site_counts = (
            pd.Series(site_cells[finite_site_mask]).value_counts()
            .reindex(per_cell.index, fill_value=0)
            .to_numpy(dtype=np.int64, copy=False)
        )
        per_cell["requested_metaspots"] = (
            _target_metaspot_counts_from_site_counts(
                finite_site_counts,
                target_sites_per_metaspot=int(target_sites_per_metaspot),
                min_metaspots_per_cell=min_metaspots_per_cell,
                max_metaspots_per_cell=max_metaspots_per_cell,
            )
        )
    else:
        per_cell["requested_metaspots"] = np.nan
    per_cell["actual_to_requested_ratio"] = (
        per_cell["actual_metaspots"]
        / pd.to_numeric(
            per_cell["requested_metaspots"],
            errors="coerce",
        ).replace(0, np.nan)
    )
    per_cell.index = pd.Index(cell_labels).take(per_cell.index.to_numpy()).rename("cell_id")
    return per_cell.reset_index()

def _grouped_nanmean_std(
    codes: np.ndarray,
    values: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute NaN-aware population moments for contiguous integer groups."""
    valid = np.isfinite(values)
    counts = np.bincount(codes[valid], minlength=n_groups).astype(
        np.int64,
        copy=False,
    )
    sums = np.bincount(
        codes[valid],
        weights=values[valid],
        minlength=n_groups,
    )
    sumsq = np.bincount(
        codes[valid],
        weights=np.square(values[valid], dtype=np.float64),
        minlength=n_groups,
    )
    mean = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    variance = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(sumsq, counts, out=variance, where=counts > 0)
    variance -= np.square(mean)
    variance = np.maximum(variance, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

def _grouped_geometry_shape(
    codes: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_groups: int,
    *,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sample-covariance elongation and spread for each MetaSpot."""
    x_mean, _ = _grouped_nanmean_std(codes, x_values, n_groups)
    y_mean, _ = _grouped_nanmean_std(codes, y_values, n_groups)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    counts = np.bincount(codes[valid], minlength=n_groups).astype(
        np.int64,
        copy=False,
    )
    x = x_values[valid].astype(np.float64, copy=False)
    y = y_values[valid].astype(np.float64, copy=False)
    grouped = lambda weights: np.bincount(
        codes[valid], weights=weights, minlength=n_groups
    )
    sum_x = grouped(x)
    sum_y = grouped(y)
    sum_x2 = grouped(np.square(x))
    sum_y2 = grouped(np.square(y))
    sum_xy = grouped(x * y)
    mx = x_mean.astype(np.float64, copy=False)
    my = y_mean.astype(np.float64, copy=False)
    count_float = counts.astype(np.float64, copy=False)
    var_x = sum_x2 - 2.0 * mx * sum_x + count_float * np.square(mx)
    var_y = sum_y2 - 2.0 * my * sum_y + count_float * np.square(my)
    cov_xy = sum_xy - mx * sum_y - my * sum_x + count_float * mx * my
    denominator = np.maximum(count_float - 1.0, 1.0)
    var_x /= denominator
    var_y /= denominator
    cov_xy /= denominator
    trace = np.maximum(var_x + var_y, 0.0)
    discriminant = np.maximum(
        np.square(trace) - 4.0 * (var_x * var_y - np.square(cov_xy)),
        0.0,
    )
    root = np.sqrt(discriminant)
    lambda_1 = np.maximum(0.5 * (trace + root), eps)
    lambda_2 = np.maximum(0.5 * (trace - root), eps)
    elongation = np.full(n_groups, np.nan, dtype=np.float32)
    spread = np.full(n_groups, np.nan, dtype=np.float32)
    valid_shape = counts >= 3
    elongation[valid_shape] = (
        lambda_1[valid_shape] / lambda_2[valid_shape]
    ).astype(np.float32)
    spread[valid_shape] = np.sqrt(trace[valid_shape]).astype(np.float32)
    return elongation, spread

def build_adata1(point_df: pd.DataFrame, show_internal_progress: bool = False) -> ad.AnnData:
    metaspot_cat = pd.Categorical(point_df[METASPOT_COL], ordered=False)
    row_codes = metaspot_cat.codes.astype(np.int64)
    row_cats = metaspot_cat.categories
    gene_codes, gene_uni = pd.factorize(point_df['feature_name'], sort=False)
    gene_codes = gene_codes.astype(np.int64)
    X = sp.coo_matrix(
        (np.ones(len(point_df), dtype=np.int32), (row_codes, gene_codes)),
        shape=(len(row_cats), len(gene_uni))
    ).tocsr()
    del gene_codes
    adata1 = ad.AnnData(X=X, obs=pd.DataFrame(index=row_cats), var=pd.DataFrame(index=gene_uni))

    n_groups = len(row_cats)
    # Do not materialize a five-column, row-sorted transcript array.  The
    # COAD run contains hundreds of millions of rows, so that copy alone was
    # several GiB before any per-MetaSpot statistics were calculated.
    codes = row_codes.astype(
        np.int32 if n_groups <= np.iinfo(np.int32).max else np.int64,
        copy=False,
    )
    if codes is not row_codes:
        del row_codes
    counts = np.bincount(codes, minlength=n_groups).astype(np.int64, copy=False)

    r_values = point_df['r_signed'].to_numpy(dtype=np.float32, copy=False)
    r_mean, r_std = _grouped_nanmean_std(codes, r_values, n_groups)
    # Pandas executes grouped quantiles in native code.  This preserves the
    # exact NumPy interpolation rule used previously while avoiding one Python
    # percentile call per MetaSpot.
    quantiles = (
        pd.Series(r_values, copy=False)
        .groupby(codes, sort=True, observed=True)
        .quantile([0.1, 0.5, 0.9])
        .to_numpy(dtype=np.float32, copy=False)
        .reshape(n_groups, 3)
    )
    r_p10, r_p50, r_p90 = quantiles.T
    del quantiles, r_values

    d1_values = point_df['d_cell_norm'].to_numpy(dtype=np.float32, copy=False)
    d1_mean, _ = _grouped_nanmean_std(codes, d1_values, n_groups)
    del d1_values
    d2_values = point_df['d_nuc_norm'].to_numpy(dtype=np.float32, copy=False)
    d2_mean, _ = _grouped_nanmean_std(codes, d2_values, n_groups)
    del d2_values
    x_values = point_df['x_location'].to_numpy(dtype=np.float32, copy=False)
    y_values = point_df['y_location'].to_numpy(dtype=np.float32, copy=False)
    elong, spread = _grouped_geometry_shape(
        codes,
        x_values,
        y_values,
        n_groups,
        eps=1e-8,
    )
    del x_values, y_values
    log_n = np.log1p(counts).astype(np.float32)

    md_stats = pd.DataFrame({
        METASPOT_COL: row_cats,
        'r_mean': r_mean, 'r_std': r_std,
        'r_p10': r_p10, 'r_p50': r_p50, 'r_p90': r_p90,
        'd_cell_norm_mean': d1_mean, 'd_nuc_norm_mean': d2_mean,
        'elongation': elong, 'spread': spread, 'log_n': log_n
    }).set_index(METASPOT_COL)
    adata1.obs = adata1.obs.join(md_stats, how='left')
    emb_cols = [c for c in point_df.columns if c.startswith('comp_emb')]
    if len(emb_cols):
        agg_comp = (
            point_df.groupby(METASPOT_COL)[emb_cols + ['local_density']]
            .agg(['mean', 'std'])
        )
        agg_comp.columns = [f"{a}_{b}" for a, b in agg_comp.columns]
        adata1.obs = adata1.obs.join(agg_comp, how='left')
    return adata1

def _validate_adata1_point_df_alignment(
    adata1: ad.AnnData,
    point_df: pd.DataFrame,
    *,
    geometry_atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Verify that a cached Module-4 object matches the current MetaSpots.

    MetaSpot identifiers are local cluster numbers (for example ``cell:17``)
    and can be reassigned when Module 3 is recomputed.  Set equality alone is
    therefore insufficient: the transcript counts and geometry attached to
    every identifier must also agree before a cached ``adata1`` can be reused.
    """
    if METASPOT_COL not in point_df.columns:
        return False, f"point_df is missing {METASPOT_COL!r}"
    if not adata1.obs_names.is_unique:
        return False, "cached adata1 has duplicate MetaSpot identifiers"
    if len(point_df) == 0:
        return (
            (adata1.n_obs == 0),
            "empty point_df" if adata1.n_obs == 0 else "point_df is empty",
        )

    # One factorisation and bincounts instead of string groupbys (1e8+ rows).
    metaspot_ids = point_df[METASPOT_COL]
    if metaspot_ids.isna().any():
        metaspot_ids = metaspot_ids.astype(str)
    spot_codes, spot_values = pd.factorize(metaspot_ids)
    point_ids = pd.Index(spot_values).astype(str)
    point_counts = pd.Series(
        np.bincount(spot_codes, minlength=len(point_ids)).astype(np.int64),
        index=point_ids,
    )
    cached_ids = pd.Index(adata1.obs_names.astype(str))
    missing = point_ids.difference(cached_ids)
    extra = cached_ids.difference(point_ids)
    if len(missing) or len(extra):
        return (
            False,
            "MetaSpot identifier mismatch "
            f"(missing_in_cache={len(missing)}, extra_in_cache={len(extra)})",
        )

    point_counts.index = point_ids
    expected_counts = point_counts.reindex(cached_ids).to_numpy(np.int64)
    cached_counts = np.asarray(adata1.X.sum(axis=1)).ravel()
    if cached_counts.shape != expected_counts.shape or not np.allclose(
        cached_counts,
        expected_counts,
        rtol=0.0,
        atol=0.0,
    ):
        mismatched = int(
            np.count_nonzero(
                ~np.isclose(
                    cached_counts,
                    expected_counts,
                    rtol=0.0,
                    atol=0.0,
                )
            )
        )
        return False, f"per-MetaSpot transcript counts differ ({mismatched} rows)"

    if "r_signed" in point_df.columns and "r_mean" in adata1.obs.columns:
        r_values = pd.to_numeric(point_df["r_signed"], errors="coerce").to_numpy(dtype=float)
        present = ~np.isnan(r_values)
        r_sum = np.bincount(spot_codes[present], weights=r_values[present], minlength=len(point_ids))
        r_n = np.bincount(spot_codes[present], minlength=len(point_ids))
        with np.errstate(invalid="ignore", divide="ignore"):
            point_r_mean = pd.Series(np.where(r_n > 0, r_sum / np.maximum(r_n, 1), np.nan), index=point_ids)
        expected_r = point_r_mean.reindex(cached_ids).to_numpy(dtype=float)
        cached_r = pd.to_numeric(
            adata1.obs["r_mean"],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite = np.isfinite(expected_r) & np.isfinite(cached_r)
        if not np.array_equal(np.isfinite(expected_r), np.isfinite(cached_r)):
            return False, "per-MetaSpot geometry missingness differs"
        if finite.any() and not np.allclose(
            cached_r[finite],
            expected_r[finite],
            rtol=0.0,
            atol=max(float(geometry_atol), 0.0),
        ):
            max_abs = float(
                np.max(np.abs(cached_r[finite] - expected_r[finite]))
            )
            return (
                False,
                "per-MetaSpot r_mean differs "
                f"(max_abs_difference={max_abs:.6g})",
            )

    return True, "aligned"
