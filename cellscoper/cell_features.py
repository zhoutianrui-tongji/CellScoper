"""Cell x structure table (``structure_feature_table``): label shares of the structure labels and
MetaDomains, and a depth-unbiased pair co-localisation statistic weighted by genes of the algorithm's own
representations (see the block comment below)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------------
# 细胞 × 结构特征表（2026-11-03）：全部来自算法输出（结构标签、MetaDomain、基因嵌入/解码器的基因权重），不用原始标记组成。
#
# 每个细胞的每一侧（核内 / 胞质）、每个结构或 MetaDomain 一行：
#   metaspot_share / transcript_share  带这个标签的 MetaSpot / 转录本比例（"面积"和"量"；随深度变化：浅的细胞标得少）
#   pair_obs / pair_exp / concentration  成对共定位：半径 R 内不同基因转录本对的权重积之和，除以把基因名在这一侧内部
#        随机打乱时的期望（解析解）。= 1 随机摆放，> 1 聚集。随机稀释转录本时分子分母按同一比例缩小，所以没有深度偏差
#        （Atera 1k，50% / 25% 深度：超出 1 的部分保留约 100%）；但单个细胞噪声大（Atera 5k 两半基因间信度 .12-.30），
#        比较细胞群体时用 pooled_concentration：sum(pair_obs) / sum(pair_exp)。
# ---------------------------------------------------------------------------------------------------------------------

_SIDE_KEYS = ["cell_id", "in_nucleus"]
STRUCTURE_SIDE = {"ER": False, "OMM": False, "Nuclear Speckle": True, "Nucleolus": True, "Lamina": True}


def ge_structure_weights(ge_vectors: np.ndarray, ge_genes: Sequence[str], marker_sets: Mapping[str, Sequence[str]],
                         structures: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Gene weight of each structure for the pair statistic: the non-negative loading of every gene on the structure's
    direction in the spatial gene embedding (mean vector of its markers minus the mean of all genes, unit length).
    Every gene with a positive loading contributes, so the statistic does not rest on the marker genes alone.
    Returns genes (upper case) x structures; structures with no marker in the embedding are left out."""
    vec = np.asarray(ge_vectors, np.float64)
    vec = vec - vec.mean(0)
    genes = pd.Index(pd.Series(np.asarray(ge_genes)).astype(str).str.upper())
    out = {}
    for s in (structures or marker_sets.keys()):
        use = genes.isin({str(g).upper() for g in marker_sets.get(s, ())})
        if not use.any():
            continue
        d = vec[use].mean(0)
        out[s] = np.clip(vec @ (d / max(np.linalg.norm(d), 1e-12)), 0, None)
    return pd.DataFrame(out, index=genes)


def metadomain_gene_weights(profiles: np.ndarray, genes: Sequence[str], names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Gene weight of each MetaDomain: the positive part of its decoder profile. Columns ``names`` (MD0, MD1...)."""
    profiles = np.atleast_2d(np.asarray(profiles, np.float64))
    names = list(names) if names is not None else [f"MD{k}" for k in range(len(profiles))]
    idx = pd.Index(pd.Series(np.asarray(genes)).astype(str).str.upper())
    return pd.DataFrame(np.clip(np.nan_to_num(profiles, nan=0.0), 0, None).T, index=idx, columns=names)


def _codes_as_text(values) -> Tuple[np.ndarray, np.ndarray]:
    """(每行整数编号, 编号对应的文字)：只对唯一值做 str 转换；缺失值编号 -1。"""
    codes, uniques = pd.factorize(values)
    return codes, np.asarray(pd.Index(uniques).astype(str), dtype=object)


def _cell_codes(points: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    codes, text = _codes_as_text(points["cell_id"])
    if (codes < 0).any():
        raise ValueError("cell_id has missing values")
    return codes, text


def label_shares(points: pd.DataFrame, label_col: str, features: Sequence[str], *, metaspot_col: str = "metaspot",
                 side_of: Optional[Mapping[str, Optional[bool]]] = None, labels=None) -> pd.DataFrame:
    """Per cell side and feature (a value of ``label_col``): share of the side's MetaSpots and transcripts carrying it.
    Sides without the label get 0. ``side_of[feature]`` restricts a feature to one side (None or missing: both).
    ``labels``: per-row label values used instead of ``points[label_col]``. A MetaSpot counts with the label of its first row."""
    # 全程用整数编号计数（几亿行时 str 转换和字符串分组是主要耗时）；细胞侧按 (cell_id 文字, in_nucleus) 排序，
    # 与原先 groupby 的行序一致。
    cell_codes, cell_text = _cell_codes(points)
    order = np.argsort(cell_text, kind="stable")
    cell_rank = np.empty(len(cell_text), np.int64)
    cell_rank[order] = np.arange(len(cell_text))
    side_key = cell_rank[cell_codes] * 2 + points["in_nucleus"].astype(bool).to_numpy()
    present = np.flatnonzero(np.bincount(side_key))  # 编号有界，计数代替排序
    n_sides = len(present)
    lookup = np.full(int(present[-1]) + 1 if n_sides else 0, -1, np.int64)
    lookup[present] = np.arange(n_sides)
    row_side = lookup[side_key]
    spot_codes = pd.factorize(points[metaspot_col])[0].astype(np.int64)
    first = np.flatnonzero(~pd.Index(row_side * (spot_codes.max() + 1) + spot_codes).duplicated())
    label_codes, label_text = _codes_as_text(points[label_col] if labels is None else labels)
    base = pd.DataFrame({"cell_id": cell_text[order][present // 2], "in_nucleus": present % 2 == 1,
                         "n_transcripts": np.bincount(row_side, minlength=n_sides),
                         "n_metaspots": np.bincount(row_side[first], minlength=n_sides)})
    parts = []
    for feat in features:
        hit = np.flatnonzero(label_text == str(feat))
        code = hit[0] if hit.size else -2  # -2：没有这个标签，计数为 0
        tx = np.bincount(row_side[label_codes == code], minlength=n_sides).astype(np.float64)
        sp = np.bincount(row_side[first][label_codes[first] == code], minlength=n_sides).astype(np.float64)
        side = (side_of or {}).get(feat)
        keep = np.ones(n_sides, bool) if side is None else (base.in_nucleus.to_numpy() == side)
        b = base[keep]
        parts.append(b.assign(feature=str(feat), metaspot_share=sp[keep] / b.n_metaspots.to_numpy(),
                              transcript_share=tx[keep] / b.n_transcripts.to_numpy()))
    return pd.concat(parts, ignore_index=True) if parts else base.iloc[:0]


def _pair_parts_one_side(xyz: np.ndarray, code: np.ndarray, weights: np.ndarray, radius: float) -> Tuple[int, np.ndarray, np.ndarray]:
    """Neighbour pairs, observed sum of w_i w_j over pairs of different genes, and its expectation under gene shuffling."""
    from scipy.spatial import cKDTree

    n = len(code)
    local, inverse = np.unique(code, return_inverse=True)
    w = weights[local]                                   # genes of this side x features
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    i, j = inverse[pairs[:, 0]], inverse[pairs[:, 1]]
    diff = i != j
    obs = (w[i[diff]] * w[j[diff]]).sum(0)
    cw = np.bincount(inverse, minlength=len(local)).astype(np.float64)[:, None] * w
    exp = len(pairs) * (cw.sum(0) ** 2 - (cw ** 2).sum(0)) / (n * (n - 1))
    return len(pairs), obs, exp


def _pair_parts_block(block_keys, block_xyz, block_codes, W, radius):
    """一批细胞侧的成对统计；参数只含这批细胞的数据，可以跨进程发送。"""
    return [(k, *_pair_parts_one_side(x, c, W, radius)) for k, x, c in zip(block_keys, block_xyz, block_codes)]


def pair_concentration(points: pd.DataFrame, gene_weights: pd.DataFrame, radius: float, *,
                       side_of: Optional[Mapping[str, Optional[bool]]] = None, min_transcripts: int = 20,
                       n_jobs: int = 1) -> pd.DataFrame:
    """Pair co-localisation of every feature (column of ``gene_weights``) in every cell side with at least
    ``min_transcripts`` transcripts. Positions: x/y and z when present (same units, ``radius`` in them)."""
    from joblib import Parallel, delayed

    axes = ["x_location", "y_location"] + (["z_location"] if "z_location" in points.columns and points["z_location"].notna().all() else [])
    xyz = points[axes].to_numpy(np.float64)
    gene_col = points["feature_name"]
    if isinstance(gene_col.dtype, pd.CategoricalDtype):  # 只转换类别一次，不为每条转录本生成新字符串
        upper = gene_col.cat.categories.astype(str).str.upper().to_numpy(dtype=object)
        codes, genes = pd.factorize(upper[gene_col.cat.codes.to_numpy()])
    else:
        codes, genes = pd.factorize(gene_col.astype(str).str.upper())
    W = gene_weights.reindex(pd.Index(genes)).fillna(0.0).to_numpy(np.float64)
    feats = list(gene_weights.columns)
    # 每个细胞侧的行号（升序）；用整数编号分组，不做逐行 str 转换。侧的先后只影响计算批次，不影响结果
    cell_codes, cell_text = _cell_codes(points)
    side_key = cell_codes.astype(np.int64) * 2 + points["in_nucleus"].astype(bool).to_numpy()
    by_side = np.argsort(side_key, kind="stable")
    cut = np.flatnonzero(np.diff(side_key[by_side])) + 1
    sides = {(cell_text[k // 2], bool(k % 2)): idx for k, idx in zip(side_key[by_side[np.r_[0, cut]]], np.split(by_side, cut))} if len(by_side) else {}
    keys = [k for k, idx in sides.items() if len(idx) >= min_transcripts]

    blocks = [keys[s:s + 256] for s in range(0, len(keys), 256)]
    # 每批只发送这批细胞的切片；不能用闭包，闭包会把整个 xyz 和 sides 序列化后随每个任务发送一次
    tasks = ((b, [xyz[sides[k]] for k in b], [codes[sides[k]] for k in b]) for b in blocks)
    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(_pair_parts_block)(b, x, c, W, radius) for b, x, c in tasks)
    else:
        results = [_pair_parts_block(b, x, c, W, radius) for b, x, c in tasks]
    rows = []
    for block in results:
        for (cell, side), n_pairs, obs, exp in block:
            n = len(sides[(cell, side)])
            for f, o, e in zip(feats, obs, exp):
                want = (side_of or {}).get(f)
                if want is None or want == side:
                    rows.append((cell, side, f, n, n_pairs, o, e))
    out = pd.DataFrame(rows, columns=["cell_id", "in_nucleus", "feature", "n_transcripts", "pairs", "pair_obs", "pair_exp"])
    out["concentration"] = out.pair_obs / out.pair_exp.where(out.pair_exp > 0)
    return out


def structure_feature_table(points: pd.DataFrame, *, structure_weights: pd.DataFrame, radius: float,
                            label_col: str = "subcellular_annotation", domain_col: Optional[str] = None,
                            domain_weights: Optional[pd.DataFrame] = None, metaspot_col: str = "metaspot",
                            min_transcripts: int = 20, n_jobs: int = 1) -> pd.DataFrame:
    """Cell x structure table (long format): one row per cell side and feature, ``kind`` = structure (labels of
    ``label_col``, on the structure's own side) or metadomain (``domain_col`` ids >= 0 named MD<id>, on both sides).
    Columns: n_transcripts, n_metaspots, metaspot_share, transcript_share, pairs, pair_obs, pair_exp, concentration."""
    structs = [s for s in structure_weights.columns]
    side_of: Dict[str, Optional[bool]] = {s: STRUCTURE_SIDE.get(s) for s in structs}
    shares = [label_shares(points, label_col, structs, metaspot_col=metaspot_col, side_of=side_of).assign(kind="structure")]
    weights = [structure_weights]
    if domain_col is not None and domain_weights is not None and domain_weights.shape[1]:
        # 只对唯一值做数字转换和命名，再按编号展开到每行
        codes, uniques = pd.factorize(points[domain_col])
        dom = pd.to_numeric(pd.Series(np.asarray(uniques, dtype=object)), errors="coerce").fillna(-1).astype(int).to_numpy()
        names = np.append(np.where(dom >= 0, np.char.add("MD", dom.astype(str)), "background").astype(object), "background")
        shares.append(label_shares(points, domain_col, list(domain_weights.columns), metaspot_col=metaspot_col,
                                   labels=names[codes]).assign(kind="metadomain"))
        weights.append(domain_weights)
    all_w = pd.concat(weights, axis=1).fillna(0.0)
    pairs = pair_concentration(points, all_w, radius, side_of=side_of, min_transcripts=min_transcripts, n_jobs=n_jobs)
    table = pd.concat(shares, ignore_index=True)
    table = table.merge(pairs.drop(columns=["n_transcripts"]), on=["cell_id", "in_nucleus", "feature"], how="left")
    return table[["cell_id", "in_nucleus", "kind", "feature", "n_transcripts", "n_metaspots", "metaspot_share",
                  "transcript_share", "pairs", "pair_obs", "pair_exp", "concentration"]]


def pooled_concentration(table: pd.DataFrame, groups: pd.Series, *, min_cells: int = 20) -> pd.DataFrame:
    """Group-level concentration: sum(pair_obs) / sum(pair_exp) over the cells of each group (``groups``: cell_id ->
    group), per feature and side, with the mean metaspot_share. Groups with fewer than ``min_cells`` cells are dropped."""
    t = table.assign(group=table["cell_id"].astype(str).map(groups.astype(str))).dropna(subset=["group"])
    keys = ["group"] + [c for c in ("kind", "feature", "in_nucleus") if c in t.columns]
    agg = {"cells": ("cell_id", "nunique"), "pair_obs": ("pair_obs", "sum"), "pair_exp": ("pair_exp", "sum")}
    agg.update({c: (c, "mean") for c in ("metaspot_share", "transcript_share") if c in t.columns})
    g = t.groupby(keys).agg(**agg).reset_index()
    g["concentration"] = g.pair_obs / g.pair_exp.where(g.pair_exp > 0)
    return g[g.cells >= min_cells].reset_index(drop=True)


__all__ = ["STRUCTURE_SIDE", "ge_structure_weights", "label_shares", "metadomain_gene_weights", "pair_concentration",
            "pooled_concentration", "structure_feature_table"]
