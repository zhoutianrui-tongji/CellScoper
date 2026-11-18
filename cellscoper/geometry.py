"""Module 1: transcript geometry (distances to the cell and nucleus boundaries, nucleus membership, signed radial
position and direction)."""

from typing import List, Optional
from matplotlib.path import Path
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd

from .runtime import _preferred_n_jobs, _progress_iter


def _stretch_signed_geometry(values: np.ndarray, cell_size: int) -> np.ndarray:
    """Robustly scale signed geometry while keeping the nuclear boundary at zero."""
    x = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return x

    if cell_size >= 16:
        negative = finite & (x < 0)
        positive = finite & (x > 0)
        if negative.any():
            neg_scale = float(np.nanpercentile(np.abs(x[negative]), 95.0))
            if np.isfinite(neg_scale) and neg_scale > 1e-6:
                x[negative] /= neg_scale
        if positive.any():
            pos_scale = float(np.nanpercentile(x[positive], 95.0))
            if np.isfinite(pos_scale) and pos_scale > 1e-6:
                x[positive] /= pos_scale

    x = np.clip(x, -2, 2)
    adaptive_band = 0.05 * (1.0 + 0.5 * np.exp(-cell_size / 1000))
    x[np.abs(x) < adaptive_band] = 0.0
    return x.astype(np.float32, copy=False)

def _point_to_ring_distance(
    points: np.ndarray,
    ring: np.ndarray,
    chunk_size: int = 20000,
) -> np.ndarray:
    """Exact distance from each point to the edges of a closed polygon ring."""
    pts = np.asarray(points, dtype=np.float64)
    start = np.asarray(ring, dtype=np.float64)
    end = np.roll(start, -1, axis=0)
    edge = end - start
    edge_sq = np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-24)
    out = np.empty(pts.shape[0], dtype=np.float64)
    for lo in range(0, pts.shape[0], chunk_size):
        rel = pts[lo:lo + chunk_size, None, :] - start[None, :, :]
        t = np.clip(np.einsum("nmj,mj->nm", rel, edge) / edge_sq, 0.0, 1.0)
        gap = rel - t[:, :, None] * edge[None, :, :]
        out[lo:lo + chunk_size] = np.sqrt(
            np.einsum("nmj,nmj->nm", gap, gap).min(axis=1)
        )
    return out

def _ordered_polygon(pts: np.ndarray) -> Optional[np.ndarray]:
    """One cell's boundary vertices (float64, NaN rows dropped) as an ordered polygon, or None.

    Apparently ordered contours are kept; clearly unordered ones (or invalid
    Shapely polygons) are rebuilt as a convex hull.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError

    # ``np.unique(..., axis=0)`` sorts rows, which destroys contour
    # order. Keep the first occurrence of each vertex instead.
    if len(pts):
        _, first_idx = np.unique(pts, axis=0, return_index=True)
        pts = pts[np.sort(first_idx)]

    if len(pts) < 3:
        return None

    try:
        tree = KDTree(pts)
        nn_dist, _ = tree.query(pts, k=2, return_distance=True)
        local_scale = float(np.nanmedian(nn_dist[:, 1]))
    except Exception:
        local_scale = np.nan

    closed_pts = np.vstack([pts, pts[:1]])
    path_steps = np.sqrt(
        np.sum(np.diff(closed_pts, axis=0) ** 2, axis=1)
    )
    typical_step = float(np.nanmedian(path_steps))
    upper_step = float(np.nanpercentile(path_steps, 95))

    ordered = (
        np.isfinite(local_scale)
        and local_scale > 0
        and typical_step <= 3.0 * local_scale
        and upper_step <= 10.0 * local_scale
    )

    polygon_pts = pts

    # A short vertex list can pass the step-size heuristic while still
    # being self-intersecting (for example, bow-tie ordering). Treat
    # only a valid, non-zero-area Shapely polygon as an ordered contour.
    if ordered:
        try:
            from shapely.geometry import Polygon as _Polygon

            candidate = _Polygon(pts)
            ordered = bool(
                candidate.is_valid
                and not candidate.is_empty
                and float(candidate.area) > 0.0
            )
        except Exception:
            ordered = False

    if not ordered:
        try:
            hull = ConvexHull(pts)
            polygon_pts = pts[hull.vertices]
        except QhullError:
            center = np.nanmedian(pts, axis=0)
            angle = np.arctan2(
                pts[:, 1] - center[1],
                pts[:, 0] - center[0],
            )
            polygon_pts = pts[np.argsort(angle)]

    return polygon_pts if len(polygon_pts) >= 3 else None

def _cell_nucleus_flags(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside the cell's nucleus polygon (False without one)."""
    polygon = _ordered_polygon(vertices)
    if polygon is None:
        return np.zeros(len(points), dtype=bool)
    # Repeat the first vertex: with closed=True Matplotlib drops the last one.
    return Path(np.vstack([polygon, polygon[:1]]), closed=True).contains_points(points)

def build_point_geometry_v2(
    transcripts_df: pd.DataFrame,
    nucleus_boundaries_df: pd.DataFrame,
    cell_boundaries_df: pd.DataFrame,
    sampled_cells: List,
    include_directional: bool = True,
    use_input_nucleus_flags: bool = True,
    winsor_perc=(1, 99),
    show_internal_progress: bool = False,
    boundary_distance: str = "vertex",
) -> pd.DataFrame:
    DTYPE = np.float32
    boundary_distance = str(boundary_distance or "vertex").strip().lower()
    if boundary_distance not in {"vertex", "edge"}:
        raise ValueError("module1.boundary_distance must be 'vertex' or 'edge'")
    use_edge_distance = boundary_distance == "edge"

    def _group_chunks(df: pd.DataFrame, chunk_size: int):
        buf = []
        for pair in df.groupby('cell_id', sort=False):
            buf.append(pair)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _winsorize_inplace(df: pd.DataFrame, cols, perc=(1, 99)) -> None:
        lo_p, hi_p = perc
        for c in cols:
            x = df[c].to_numpy()
            lo, hi = np.nanpercentile(x, [lo_p, hi_p])
            df[c] = np.clip(x, lo, hi).astype(DTYPE, copy=False)

    def _build_kdtrees(boundary_df: pd.DataFrame, cells, desc="KDTree") -> dict:
        kd = {}
        sub = boundary_df[boundary_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(it, desc=f'Building {desc}', total=it.ngroups, show=show_internal_progress)
        for cid, g in iterator:
            coords = g[['vertex_x', 'vertex_y']].to_numpy()
            kd[cid] = KDTree(coords) if len(coords) else None
        return kd

    def _build_ordered_polygons(boundary_df: pd.DataFrame, cells, desc: str) -> dict:
        """Build valid ordered polygons from boundary-point tables.

        Some spatial formats store boundary pixels/vertices without contour
        ordering. ``matplotlib.path.Path`` assumes ordered polygon vertices;
        feeding an unordered point cloud can collapse the inferred nuclear
        area and severely reduce nuclear recall. We preserve apparently
        ordered contours and reconstruct clearly unordered contours with a
        convex hull. Nuclei are typically close to convex, making this a
        conservative and stable fallback for cross-platform input.
        """

        polygons = {}
        sub = boundary_df[boundary_df['cell_id'].isin(cells)]
        it = sub.groupby('cell_id', sort=False)
        iterator = _progress_iter(
            it,
            desc=desc,
            total=it.ngroups,
            show=show_internal_progress,
        )

        for cid, g in iterator:
            pts = (
                g[['vertex_x', 'vertex_y']]
                .apply(pd.to_numeric, errors='coerce')
                .dropna()
                .to_numpy(dtype=np.float64)
            )
            polygons[cid] = _ordered_polygon(pts)

        return polygons

    def _polygon_paths(polygons: dict) -> dict:
        return {
            # With ``closed=True`` Matplotlib treats the last vertex as a
            # CLOSEPOLY placeholder. Repeat the first point explicitly so
            # the final real vertex is not discarded.
            cid: (
                Path(np.vstack([poly, poly[:1]]), closed=True)
                if poly is not None
                else None
            )
            for cid, poly in polygons.items()
        }

    def _boundary_distance(q: np.ndarray, kt, polygon) -> np.ndarray:
        if use_edge_distance and polygon is not None:
            return _point_to_ring_distance(q, polygon).astype(DTYPE)
        d, _ = kt.query(q, k=1, return_distance=True)
        return d[:, 0].astype(DTYPE)

    # Avoid copying the full transcript table. We keep the original rows and
    # build geometry columns in chunks so Module 1 can run without doubling RAM.
    def _nearest_vertex_distance(points: pd.DataFrame, boundary_df: pd.DataFrame) -> np.ndarray:
        """Distance to the nearest boundary vertex of the point's own cell (NaN without one).

        One KD-tree over all cells: a third coordinate of cell code x a spacing
        wider than the whole slide keeps every nearest vertex inside the cell, so
        the result equals one tree per cell without 1e5 per-cell trees and queries.
        """
        from scipy.spatial import cKDTree

        cells_index = pd.Index(pd.unique(points['cell_id']))
        point_cell = cells_index.get_indexer(points['cell_id'])
        vertex_cell = cells_index.get_indexer(boundary_df['cell_id'])
        vx = pd.to_numeric(boundary_df['vertex_x'], errors='coerce').to_numpy(np.float64)
        vy = pd.to_numeric(boundary_df['vertex_y'], errors='coerce').to_numpy(np.float64)
        keep = (vertex_cell >= 0) & np.isfinite(vx) & np.isfinite(vy)
        vx, vy, vertex_cell = vx[keep], vy[keep], vertex_cell[keep]
        px = points['x_location'].to_numpy(np.float64)
        py = points['y_location'].to_numpy(np.float64)
        out = np.full(len(points), np.nan, dtype=DTYPE)
        if not len(vx):
            return out
        lo = np.nanmin([vx.min(), vy.min(), np.nanmin(px), np.nanmin(py)])
        hi = np.nanmax([vx.max(), vy.max(), np.nanmax(px), np.nanmax(py)])
        spacing = 4.0 * (hi - lo) + 1.0
        tree = cKDTree(np.c_[vx, vy, vertex_cell * spacing])
        _, nearest = tree.query(np.c_[px, py, point_cell * spacing], k=1, workers=_preferred_n_jobs())
        found = vertex_cell[nearest] == point_cell
        out[found] = np.sqrt((px[found] - vx[nearest[found]]) ** 2 + (py[found] - vy[nearest[found]]) ** 2).astype(DTYPE)
        return out

    def _per_cell_boundary_geometry(df: pd.DataFrame) -> pd.DataFrame:
        cell_kd = _build_kdtrees(cell_boundaries_df, sampled_cells, desc="Cell KDTree")
        nuc_kd = _build_kdtrees(nucleus_boundaries_df, sampled_cells, desc="Nucleus KDTree")
        need_nuc_polygons = use_edge_distance or not use_input_nucleus_flags
        nuc_polygons = (
            _build_ordered_polygons(
                nucleus_boundaries_df, sampled_cells, 'Build nucleus polygons'
            )
            if need_nuc_polygons
            else {}
        )
        cell_polygons = (
            _build_ordered_polygons(
                cell_boundaries_df, sampled_cells, 'Build cell polygons'
            )
            if use_edge_distance
            else {}
        )
        nuc_paths = {} if use_input_nucleus_flags else _polygon_paths(nuc_polygons)

        # Process grouped rows in chunks to avoid materializing several full-size
        # temporary Series at once.
        d_cell_parts = []
        d_nuc_parts = []
        flags = []
        group_iter = _group_chunks(df, chunk_size=128)
        total_groups = int(df['cell_id'].nunique(dropna=False)) if len(df) else 0
        iterator = _progress_iter(group_iter, desc='Point geometry chunks', total=max(1, (total_groups + 127) // 128), show=show_internal_progress)
        for chunk in iterator:
            for cid, g in chunk:
                kt = cell_kd.get(cid, None)
                if kt is None:
                    d_cell_parts.append(pd.Series(np.nan, index=g.index))
                else:
                    q = g[['x_location', 'y_location']].to_numpy()
                    d_cell_parts.append(pd.Series(
                        _boundary_distance(q, kt, cell_polygons.get(cid)),
                        index=g.index,
                    ))

                kt = nuc_kd.get(cid, None)
                if kt is None:
                    d_nuc_parts.append(pd.Series(np.nan, index=g.index))
                else:
                    q = g[['x_location', 'y_location']].to_numpy()
                    d_nuc_parts.append(pd.Series(
                        _boundary_distance(q, kt, nuc_polygons.get(cid)),
                        index=g.index,
                    ))

                if use_input_nucleus_flags:
                    input_flags = (
                        pd.to_numeric(g["overlaps_nucleus"], errors="coerce")
                        .fillna(0)
                        .ne(0)
                    )
                    flags.append(pd.Series(input_flags.to_numpy(), index=g.index))
                else:
                    p = nuc_paths.get(cid, None)
                    if p is None:
                        flags.append(pd.Series(False, index=g.index))
                    else:
                        pts = g[['x_location', 'y_location']].to_numpy()
                        flags.append(pd.Series(p.contains_points(pts), index=g.index))

        df['d_cell'] = pd.concat(d_cell_parts)
        df['d_nuc'] = pd.concat(d_nuc_parts)
        df['in_nucleus'] = pd.concat(flags)
        return df

    def _parallel_nucleus_flags(points: pd.DataFrame) -> np.ndarray:
        from joblib import Parallel, delayed

        point_cell, cells_index = pd.factorize(points['cell_id'])
        sub = nucleus_boundaries_df[nucleus_boundaries_df['cell_id'].isin(sampled_cells)]
        vertex_cell = pd.Index(cells_index).get_indexer(sub['cell_id'])
        vertices = sub[['vertex_x', 'vertex_y']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
        keep = (vertex_cell >= 0) & ~np.isnan(vertices).any(axis=1)
        vertex_cell, vertices = vertex_cell[keep], vertices[keep]
        vertex_order = np.argsort(vertex_cell, kind='stable')
        vertex_bounds = np.searchsorted(vertex_cell[vertex_order], np.arange(len(cells_index) + 1))
        point_order = np.argsort(point_cell, kind='stable')
        point_bounds = np.searchsorted(point_cell[point_order], np.arange(len(cells_index) + 1))
        xy = points[['x_location', 'y_location']].to_numpy()
        flags = np.zeros(len(points), dtype=bool)
        cells_with_nucleus = np.flatnonzero(np.diff(vertex_bounds) > 0)
        results = Parallel(n_jobs=_preferred_n_jobs(), batch_size=256)(
            delayed(_cell_nucleus_flags)(
                vertices[vertex_order[vertex_bounds[c]:vertex_bounds[c + 1]]],
                xy[point_order[point_bounds[c]:point_bounds[c + 1]]],
            )
            for c in cells_with_nucleus
        )
        for c, cell_flags in zip(cells_with_nucleus, results):
            flags[point_order[point_bounds[c]:point_bounds[c + 1]]] = cell_flags
        return flags

    df = transcripts_df[transcripts_df['cell_id'].isin(sampled_cells)]
    use_input_nucleus_flags = bool(
        use_input_nucleus_flags
        and "overlaps_nucleus" in df.columns
    )
    if not use_edge_distance:
        # Vectorised over all cells, nuclear polygons in parallel workers (Xenium
        # COAD, 272M transcripts in 352k cells: the per-cell loop took 36-60 min).
        df['d_cell'] = _nearest_vertex_distance(df, cell_boundaries_df)
        df['d_nuc'] = _nearest_vertex_distance(df, nucleus_boundaries_df)
        if use_input_nucleus_flags:
            df['in_nucleus'] = pd.to_numeric(df["overlaps_nucleus"], errors="coerce").fillna(0).ne(0).to_numpy()
        else:
            df['in_nucleus'] = _parallel_nucleus_flags(df)
    else:
        df = _per_cell_boundary_geometry(df)

    d_cell = df['d_cell'].to_numpy(dtype=DTYPE, copy=False)
    d_nuc = df['d_nuc'].to_numpy(dtype=DTYPE, copy=False)
    denom = (d_cell + d_nuc).astype(DTYPE, copy=False)
    denom[denom <= 0] = np.nan

    inside = df['in_nucleus'].to_numpy()
    df['r_signed'] = np.where(inside, -d_cell/denom, d_nuc/denom).astype(DTYPE, copy=False)
    df['d_cell_norm'] = (d_cell/denom).astype(DTYPE, copy=False)
    df['d_nuc_norm'] = (d_nuc/denom).astype(DTYPE, copy=False)

    if include_directional:
        nuc_cent = (nucleus_boundaries_df[nucleus_boundaries_df['cell_id'].isin(sampled_cells)]
                    .groupby('cell_id', sort=False)
                    .agg(nuc_x=('vertex_x', 'mean'), nuc_y=('vertex_y', 'mean')))
        df = df.join(nuc_cent, on='cell_id')
        nan_c = df['nuc_x'].isna() | df['nuc_y'].isna()
        if nan_c.any():
            fb = (df.loc[nan_c]
                  .groupby('cell_id')[['x_location', 'y_location']]
                  .mean()
                  .rename(columns={'x_location': 'nuc_x', 'y_location': 'nuc_y'}))
            df.update(fb, overwrite=False)
        dx = (df['x_location'] - df['nuc_x']).to_numpy(dtype=DTYPE, copy=False)
        dy = (df['y_location'] - df['nuc_y']).to_numpy(dtype=DTYPE, copy=False)
        norm = np.sqrt(dx**2 + dy**2)
        norm[norm <= 0] = np.nan
        df['cos_theta'] = (dx / norm).astype(DTYPE, copy=False)
        df['sin_theta'] = (dy / norm).astype(DTYPE, copy=False)
        st = df['sin_theta'].to_numpy(dtype=DTYPE, copy=False)
        ct = df['cos_theta'].to_numpy(dtype=DTYPE, copy=False)
        df['sin2_theta'] = (2.0 * st * ct).astype(DTYPE, copy=False)
        df['cos2_theta'] = (ct*ct - st*st).astype(DTYPE, copy=False)

    # Nuclear radius estimation and per-cell stretching
    nuc_cent_xy = (nucleus_boundaries_df.groupby('cell_id', sort=False)
                   .agg(_nx=('vertex_x', 'mean'), _ny=('vertex_y', 'mean')))
    nuc_bd = (nucleus_boundaries_df
              .join(nuc_cent_xy, on='cell_id')
              .assign(_r=lambda t: np.sqrt((t['vertex_x']-t['_nx'])**2 + (t['vertex_y']-t['_ny'])**2)))
    rad_df = nuc_bd.groupby('cell_id', sort=False)['_r'].median().rename('_nuc_r')
    df = df.join(rad_df, on='cell_id')

    _nuc_r = df['_nuc_r'].to_numpy(dtype=DTYPE, copy=False)
    _nuc_r = np.where((~np.isfinite(_nuc_r)) | (_nuc_r <= 0), np.nan, _nuc_r)
    rs_norm = df['r_signed'].to_numpy(dtype=DTYPE, copy=False) / np.nan_to_num(_nuc_r, nan=1.0)

    df['_rs_tmp'] = rs_norm.astype(DTYPE, copy=False)
    cell_sizes = df.groupby('cell_id').size()
    df['_cell_size'] = df['cell_id'].map(cell_sizes)
    # Same per-cell call on NumPy slices (rows in their original order within each cell).
    rs_values = df['_rs_tmp'].to_numpy()
    stretched = np.empty(len(df), dtype=DTYPE)
    cell_codes = pd.factorize(df['cell_id'])[0]
    by_cell = np.argsort(cell_codes, kind='stable')
    for rows in np.split(by_cell, np.flatnonzero(np.diff(cell_codes[by_cell])) + 1):
        if len(rows):
            stretched[rows] = _stretch_signed_geometry(rs_values[rows], len(rows))
    df['r_signed'] = stretched
    df.drop(columns=['_rs_tmp', '_nuc_r', '_cell_size'], errors='ignore', inplace=True)

    _winsorize_inplace(df, ['d_cell', 'd_nuc', 'r_signed', 'd_cell_norm', 'd_nuc_norm'], winsor_perc)
    return df
