"""运行诊断图：把流程每一步的初步结果画到 ``out_dir/figures/``，便于检查流程每一步做了什么。

每张图只读运行中已有的小规模汇总（adata1.obs / uns、注释报告、细胞 × 结构表）或少量示例细胞，
全量 Atera（~17 万细胞、~5 亿转录本）也只多几分钟。任何一张图失败只记警告，不影响运行结果。

图号与流程顺序一致：
01 外来转录本删除（预处理阶段写出，plot_foreign_removal）
02 MetaSpot 划分        03 基因嵌入        04 LENS 训练曲线（LENS 训练后写出，plot_lens_training）
05 LENS 信号 vs 打乱    06 MetaDomain      07 注释验证        08 标签占比 vs RNA 深度
09 示例细胞

样式：所有图共用 STYLE（只在画图时生效，不改全局 rcParams）。颜色取自参考调色板并用校验脚本检查过：
核内结构 4 色两两可分、胞质 2 色可分（色盲 ΔE ≥ 16）；MetaDomain 3 色红绿色盲 ΔE 6.9，示例细胞里
另用点形状区分。单系列用蓝色，对照（打乱基因、删除前）用灰色。
"""
from __future__ import annotations

import functools
import json
import logging
from typing import Callable, List, Mapping, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .checkpoints import _save_fig

FIGURES_DIR = "figures"
_LOG = logging.getLogger("cellscoper")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#ebeae6"
ACCENT, ACCENT_LIGHT, NULL = "#2a78d6", "#cde2fb", "#b4b2ab"
DIVERGING = LinearSegmentedColormap.from_list("cellscoper_div", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("cellscoper_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

# 同一结构在所有图里同色（核内 4 色、胞质 2 色分别两两可分；Nucleus/Cytosol 为中性灰）
STRUCTURE_COLOURS = {
    "Nucleus": "#a19f99", "Cytosol": "#dcdbd6",
    "Nucleolus": "#e87ba4", "Lamina": "#008300", "Nuclear Pore": "#eda100", "Nuclear Speckle": "#6250d6",
    "ER": "#eb6834", "OMM": "#2a78d6",
}
_FALLBACK = ["#1baf7a", "#e34948", "#52514e"]
_DOMAIN_COLOURS = ["#1baf7a", "#e34948", "#eda100", "#6250d6", "#008300", "#e87ba4"]
_DOMAIN_MARKERS = ["o", "^", "s", "D", "v", "P"]

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "patch.linewidth": 0,
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _styled(fn):
    """画图函数在 STYLE 下执行（rc_context 退出后恢复调用者的设置）。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapper


def _colour(label: str, i: int = 0) -> str:
    return STRUCTURE_COLOURS.get(str(label), _FALLBACK[i % len(_FALLBACK)])


def _header(fig: Figure, title: str, subtitle: Optional[str] = None) -> None:
    """左上角粗体标题 + 可选的一行灰色说明；记下标题区高度，_save 用它给 tight_layout 留空。"""
    height_pt = fig.get_figheight() * 72
    fig.text(0.01, 1 - 8 / height_pt, title, ha="left", va="top", fontsize=11, fontweight="bold")
    lines = subtitle.count("\n") + 1 if subtitle else 0
    if subtitle:
        fig.text(0.01, 1 - 25 / height_pt, subtitle, ha="left", va="top", fontsize=8.5, color=INK2, linespacing=1.35)
    fig._cellscoper_top = 1 - (32 + 14 * lines) / height_pt


def _figure(width: float, height: float, title: str, ncols: int = 1, subtitle: Optional[str] = None,
            **grid) -> Tuple[Figure, np.ndarray]:
    """标题（+ 说明）+ 一行 ncols 个面板，多面板时左上角带 a、b、c 编号。"""
    fig = Figure(figsize=(width, height))
    axes = np.atleast_1d(fig.subplots(1, ncols, gridspec_kw=grid or None))
    _header(fig, title, subtitle)
    if ncols > 1:
        for letter, ax in zip("abcdefgh", axes):
            _panel_letter(ax, letter)
    return fig, axes


def _panel_letter(ax, letter: str) -> None:
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 8), textcoords="offset points",
                fontsize=11, fontweight="bold", va="bottom")


def _hist(ax, values, bins, colour=ACCENT, label=None, filled=True):
    """直方图：浅色填充 + 同色细轮廓（不画柱边框）。"""
    if filled:
        ax.hist(values, bins, histtype="stepfilled", color=colour, alpha=0.18, lw=0)
    ax.hist(values, bins, histtype="step", color=colour, lw=1.3, label=label)
    ax.grid(axis="y")


def _drawn_texts(fig: Figure):
    """实际画出来的非空文字（刻度标签只取视野内的刻度）。"""
    texts = list(fig.texts)
    for legend in fig.legends:
        texts += legend.get_texts()
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    texts.append(tick.label1)
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


def text_overlaps(fig: Figure) -> List[Tuple[str, str]]:
    """渲染后两两比较文字外框，返回互相重叠的文字对（各取前 30 个字符）。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(renderer).expanded(0.98, 0.9)) for t in _drawn_texts(fig)]
    return [(a[:30], b[:30]) for i, (a, box_a) in enumerate(boxes) for b, box_b in boxes[i + 1:]
            if box_a.overlaps(box_b)]


def _save(fig: Figure, out_dir: str, name: str, bottom: float = 0.0) -> None:
    fig.tight_layout(rect=(0, bottom, 1, getattr(fig, "_cellscoper_top", 0.97)))
    overlaps = text_overlaps(fig)
    if overlaps:
        _LOG.warning("Diagnostic figure %s: %d overlapping text pairs, e.g. %s", name, len(overlaps), overlaps[:3])
    _save_fig(fig, out_dir, name, dpi=200, folder=FIGURES_DIR)


def _thousands(x, _pos=None) -> str:
    """计数的简写：21574 → 21.6k，2000 → 2k，1403600 → 1.4M。"""
    for scale, suffix in ((1e6, "M"), (1e3, "k"), (1, "")):
        if abs(x) >= scale or scale == 1:
            v = x / scale
            text = f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.1f}"
            return text.rstrip("0").rstrip(".") + suffix if "." in text else text + suffix
    return f"{x:g}"


def _end_labels(ax, items, min_gap_pt: float = 11.0) -> None:
    """线末直接标注 [(x, y, 文字)]：按高度排序后上下至少相隔 min_gap_pt，避免互相压住。"""
    if not items:
        return
    to_px = ax.transData.transform
    pts_per_px = 72.0 / ax.figure.dpi
    placed = sorted(((to_px((x, y))[1] * pts_per_px, x, y, text) for x, y, text in items))
    last = -np.inf
    for target, x, y, text in placed:
        pos = max(target, last + min_gap_pt)
        ax.annotate(text, xy=(x, y), xytext=(6, pos - target), textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
        last = pos


def _count_axis(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_thousands))


# ---------------------------------------------------------------- 01 预处理
@_styled
def plot_foreign_removal(out_dir: str, cell_codes: np.ndarray, cell_type: np.ndarray,
                         removed: np.ndarray, threshold: float) -> None:
    """每细胞转录本数（删前/删后）、每细胞删除比例、各内部细胞组的删除比例。

    cell_codes: 每条转录本的细胞编号（0..n_cells-1）；cell_type: 每个细胞的内部 k-means 类别；
    removed: 每条转录本是否被删。"""
    ok = cell_codes >= 0
    n_cells = len(cell_type)
    before = np.bincount(cell_codes[ok], minlength=n_cells)
    gone = np.bincount(cell_codes[ok], weights=removed[ok].astype(float), minlength=n_cells)
    share = gone / np.maximum(before, 1)
    total, n_all = int(gone.sum()), int(before.sum())
    fig, ax = _figure(13, 3.9, f"Module 1 · foreign transcripts removed: {total:,} of {n_all:,} "
                               f"({100 * total / max(n_all, 1):.1f}%)", 3)
    lo = max(np.log10(max(before[before > 0].min(), 1)) - 0.05, 0)
    bins = np.logspace(lo, np.log10(max(before.max(), 10)) + 0.05, 60)
    _hist(ax[0], before, bins, NULL, "before", filled=False)
    _hist(ax[0], before - gone, bins, ACCENT, "after")
    ax[0].set_xscale("log")
    ax[0].set(xlabel="transcripts per cell", ylabel="cells", title="Cell depth")
    ax[0].legend(loc="upper left")
    _count_axis(ax[0])
    _hist(ax[1], 100 * share, 60)
    med = 100 * np.median(share)
    ax[1].axvline(med, color=INK2, lw=0.8)
    ax[1].annotate(f"median {med:.1f}%", xy=(med, 1), xycoords=("data", "axes fraction"), xytext=(4, -10),
                   textcoords="offset points", color=INK2, fontsize=8, va="top")
    ax[1].set(xlabel="removed share per cell (%)", ylabel="cells", title="Removed per cell")
    _count_axis(ax[1])
    groups = pd.DataFrame({"g": cell_type, "before": before, "gone": gone}).groupby("g").sum()
    groups["n"] = pd.Series(cell_type).value_counts()
    pct = 100 * groups["gone"] / groups["before"].clip(lower=1)
    x = np.arange(len(groups))
    ax[2].bar(x, pct, width=0.72, color=ACCENT)
    ax[2].set_xticks(x, [f"{g}\n{_thousands(n)}" for g, n in zip(groups.index, groups["n"])])
    ax[2].grid(axis="y")
    ax[2].set(xlabel="internal cell group (k-means)  ·  cells", ylabel="removed share of group RNA (%)",
              title=f"By expression group (gene < {threshold:g} × its top group)")
    _save(fig, out_dir, "01_foreign_transcripts")


# ---------------------------------------------------------------- 02 MetaSpot
@_styled
def plot_metaspots(out_dir: str, obs: pd.DataFrame) -> None:
    """MetaSpot 数 / 细胞、转录本数 / MetaSpot、细胞深度 vs MetaSpot 数。需要 obs 的 cell_id 和 log_n。"""
    n = np.rint(np.expm1(obs["log_n"].to_numpy(float)))
    per_cell = pd.DataFrame({"cell": obs["cell_id"].to_numpy(), "n": n}).groupby("cell", observed=True)["n"].agg(["size", "sum"])
    fig, ax = _figure(13, 3.9, f"Module 3 · {len(n):,} MetaSpots in {len(per_cell):,} cells", 3,
                      width_ratios=[1, 1, 1.15])
    _hist(ax[0], per_cell["size"], 60)
    ax[0].set(xlabel="MetaSpots per cell", ylabel="cells", title=f"Per cell (median {np.median(per_cell['size']):.0f})")
    _count_axis(ax[0])
    q99 = int(np.ceil(np.quantile(n, 0.99)))
    _hist(ax[1], n[n <= q99], np.arange(0.5, q99 + 1.5))  # 整数计数：每个整数一格
    ax[1].set(xlabel=f"transcripts per MetaSpot (≤ 99th pct, {q99})", ylabel="MetaSpots",
              title=f"Per MetaSpot (median {np.median(n):.0f})")
    _count_axis(ax[1])
    hb = ax[2].hexbin(per_cell["sum"], per_cell["size"], xscale="log", bins="log", gridsize=45, cmap=SEQUENTIAL,
                      mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax[2], fraction=0.05, pad=0.02)
    cb.set_label("cells", color=INK2)
    cb.outline.set_visible(False)
    ax[2].set(xlabel="transcripts per cell", ylabel="MetaSpots per cell", title="Depth vs MetaSpots")
    _save(fig, out_dir, "02_metaspots")


# ---------------------------------------------------------------- 03 基因嵌入
def _dot_vs_null(ax, rows, xlabel):
    """每行一个结构：真实值（结构色圆点）对照随机基因集 ±3 SD（灰色条）。rows: (名称, 真实, 零均值, 零标准差, 着色?)。"""
    for i, (name, real, null_mean, null_sd, coloured) in enumerate(rows):
        ax.plot([null_mean - 3 * null_sd, null_mean + 3 * null_sd], [i, i], color="#e3e2dd", lw=8, solid_capstyle="round")
        ax.plot([null_mean], [i], "|", color=MUTED, ms=10, mew=1.2)
        ax.plot(real, i, "o", ms=8, color=_colour(name, i) if coloured else NULL, mec="white", mew=1.2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_xlabel(xlabel)
    ax.legend(handles=[Line2D([], [], color="#e3e2dd", lw=8, label="random gene sets (±3 SD)"),
                       Line2D([], [], ls="", marker="o", ms=7, color=MUTED, label="observed")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)


@_styled
def plot_gene_embedding(out_dir: str, ge: Mapping, markers: Mapping[str, List[str]]) -> None:
    """a：各结构 marker 集合之间的平均余弦相似度（对角线 = 集合内部）；b：集合内部相似度 vs 同样大小的随机基因集。"""
    vectors = np.asarray(ge["vectors"], float)
    genes = pd.Index(np.asarray(ge["genes"]).astype(str)).str.upper()
    centred = vectors - vectors.mean(0)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
    order = list(STRUCTURE_COLOURS)
    sets = {name: np.flatnonzero(genes.isin(pd.Index(gl).astype(str).str.upper()))
            for name, gl in sorted(markers.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)}
    sets = {k: v for k, v in sets.items() if len(v) >= 3}
    names = list(sets)
    if not names:  # 没有任何 ≥3 个基因的 marker 集合在嵌入里
        _LOG.info("Diagnostic figure 03_gene_embedding skipped: no marker set with >= 3 embedded genes")
        return
    sums = {k: unit[v].sum(0) for k, v in sets.items()}

    def mean_cos(idx):
        k = len(idx)
        total = unit[idx].sum(0)
        return float((total @ total - k) / (k * (k - 1)))

    sim = np.array([[mean_cos(sets[a]) if a == b else float(sums[a] @ sums[b] / (len(sets[a]) * len(sets[b])))
                     for b in names] for a in names])
    rng = np.random.default_rng(0)
    rows = []
    for name in names:
        null = [mean_cos(rng.choice(len(genes), len(sets[name]), replace=False)) for _ in range(20)]
        rows.append((name, sim[names.index(name), names.index(name)], float(np.mean(null)), float(np.std(null)), True))
    fig, (ax, ax2) = _figure(
        12.5, 5.2, "Module 5 · spatial gene embedding: do markers of one structure co-localise?", 2,
        subtitle="Each gene gets a vector from which genes sit near it inside cells. If this works, marker genes of one structure "
                 "should be more similar to each other\n(diagonal) than to markers of other structures (off-diagonal) "
                 "and than random genes (b). Similarity = mean cosine between gene vectors.",
        width_ratios=[1, 1.15])
    vmax = float(np.abs(sim).max())
    im = ax.imshow(sim, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(sim[i, j]) > 0.6 * vmax else INK)
    ax.set_xticks(range(len(names)), [n.replace(" ", "\n") for n in names])
    ax.set_yticks(range(len(names)), [f"{n} ({len(sets[n]):,})" for n in names])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("mean cosine similarity", color=INK2)
    ax.set_title("Between marker sets (number of markers in brackets)")
    _dot_vs_null(ax2, rows, "mean cosine similarity between a structure's markers")
    ax2.set_title("Within a marker set vs random gene sets of equal size")
    _save(fig, out_dir, "03_gene_embedding")


# ---------------------------------------------------------------- 04 LENS 训练
@_styled
def plot_lens_training(out_dir: str, table: pd.DataFrame, best_epoch: int) -> None:
    """table: epoch, train_nll, heldout_gain_nats（module5_lens_training.csv 的列）。

    a 把留出增益换成直观单位：真实被藏基因的概率比“按细胞平均组成去猜”高多少（e^gain − 1）。"""
    epoch = table["epoch"].to_numpy()
    pct = (np.exp(table["heldout_gain_nats"].to_numpy(float)) - 1) * 100
    kept = int(best_epoch) if int(best_epoch) in epoch else int(epoch[np.argmax(pct)])
    final = float(pct[np.flatnonzero(epoch == kept)[0]])
    fig, ax = _figure(
        12.5, 4.6, "Module 5 · LENS training: learning where RNA sits inside the cell", 2,
        subtitle="Task: hide 25% of the RNA in a MetaSpot and predict which genes they were from the remaining RNA and its "
                 "neighbours.\nScore (a), on cells never used for training: how much more likely the true hidden gene is "
                 "than a guess from the cell's average RNA mix.",
        width_ratios=[1.6, 1])
    a = ax[0]
    a.plot(epoch, pct, color=ACCENT, lw=1.8)
    a.fill_between(epoch, 0, pct, color=ACCENT, alpha=0.08, lw=0)
    for share in (0.5, 0.95):
        e = int(epoch[np.argmax(pct >= share * final)])
        y = float(pct[epoch == e][0])
        a.plot(e, y, "o", ms=6, color=ACCENT, mec="white", mew=1.2)
        a.annotate(f"{int(share * 100)}% of final by epoch {e}", xy=(e, y), xytext=(10, -18), textcoords="offset points",
                   fontsize=8, color=INK2, ha="left", va="top")
    a.plot(kept, final, "o", ms=9, color=ACCENT, mec="white", mew=1.5)
    a.annotate(f"kept model: epoch {kept}\n+{final:.1f}%", xy=(kept, final), xytext=(-12, 12), textcoords="offset points",
               fontsize=8.5, color=INK, ha="right", va="bottom", fontweight="bold")
    if epoch[-1] > kept:
        a.axvspan(kept, epoch[-1], color="#f0efec", lw=0)
        a.annotate("no further gain\n→ stopped", xy=(epoch[-1], 0), xycoords=("data", "axes fraction"), xytext=(-2, 6),
                   textcoords="offset points", fontsize=7.5, color=INK2, ha="right", va="bottom")
    a.set_ylim(min(0.0, float(pct.min()) * 1.15), max(pct.max(), 1e-6) * 1.3)
    a.axhline(0, color=INK2, lw=0.8)
    a.grid(axis="y")
    a.set(xlabel="epoch (one pass over a sample of training MetaSpots)", ylabel="hidden gene: % more likely\nthan from cell average",
          title="Held-out improvement — the number that matters")
    b = ax[1]
    b.plot(epoch, table["train_nll"], color=MUTED, lw=1.2)
    b.axvline(kept, color=MUTED, lw=0.8)
    b.grid(axis="y")
    b.set(xlabel="epoch", ylabel="nats per hidden transcript", title="Training loss (should go down)")
    _save(fig, out_dir, "04_lens_training")


# ---------------------------------------------------------------- 05 LENS 信号
def pick_lens_example_cell(obs: pd.DataFrame, null_spots: np.ndarray, quantile: float = 0.8) -> Optional[str]:
    """从做过基因打乱对照的细胞里，按 RNA 深度的 quantile 分位选一个示例细胞（确定性，不看结果）。"""
    null_spots = np.asarray(null_spots, np.int64)
    if not len(null_spots) or "cell_id" not in obs.columns or "log_n" not in obs.columns:
        return None
    depth = pd.Series(np.expm1(obs["log_n"].to_numpy(float)[null_spots]),
                      index=obs["cell_id"].astype(str).to_numpy()[null_spots]).groupby(level=0).sum()
    depth = depth.sort_index().sort_values(kind="stable")
    return str(depth.index[int(quantile * (len(depth) - 1))])


NUCLEAR_STRUCTURES = {"Nucleolus", "Lamina", "Nuclear Speckle", "Nuclear Pore"}


def structure_direction_stats(z: np.ndarray, lens: Mapping, markers: Mapping[str, List[str]],
                              nucleus: np.ndarray) -> List[Tuple[str, float, float]]:
    """标签实际用到的量：z 在各结构方向上的投影（方向 = marker 的解码器行均值 − 全部基因均值，同 lens_count.gene_weights）。
    在做过打乱对照的 MetaSpot 上（只取该结构所在一侧），返回 [(结构, 真实/打乱 的标准差之比, 同一 MetaSpot 真实与打乱的 Spearman ρ)]。"""
    from scipy.stats import spearmanr

    spots = np.asarray(lens.get("null_spots", ()), np.int64)
    W = np.asarray(lens.get("gene_loadings", np.zeros((0, 0))), float)
    if not len(spots) or not W.size:
        return []
    genes = pd.Index(np.asarray(lens["gene_names"]).astype(str)).str.upper()
    real, null = np.asarray(z)[spots], np.asarray(lens["null_z"], float)
    on_nucleus = np.asarray(nucleus, bool)[spots]
    order = list(STRUCTURE_COLOURS)
    out = []
    for name in sorted(markers, key=lambda k: order.index(k) if k in order else 99):
        idx = np.flatnonzero(genes.isin(pd.Index(markers[name]).astype(str).str.upper()))
        m = on_nucleus == (name in NUCLEAR_STRUCTURES)
        if len(idx) < 3 or m.sum() < 100:
            continue
        direction = W[idx].mean(0) - W.mean(0)
        pr, pn = real[m] @ direction, null[m] @ direction
        out.append((name, float(pr.std() / max(pn.std(), 1e-12)), float(spearmanr(pr, pn)[0])))
    return out


@_styled
def plot_lens_signal(out_dir: str, z: np.ndarray, lens: Mapping, spot_names: Optional[pd.Index] = None,
                     cell_rows: Optional[pd.DataFrame] = None, cell_bounds: Optional[pd.DataFrame] = None,
                     nucleus_bounds: Optional[pd.DataFrame] = None, markers: Optional[Mapping[str, List[str]]] = None,
                     nucleus: Optional[np.ndarray] = None) -> None:
    """a：强信号 MetaSpot 的比例，真实 vs 同一批细胞内打乱基因位置；b：标签实际用到的结构方向上，真实/打乱的波动之比
    与同一 MetaSpot 真实-打乱相关（越低说明越少来自 MetaSpot 大小/位置）；c：示例细胞每个 MetaSpot 的“真实 − 打乱”信号。

    强信号 = |z| 超过打乱细胞 |z| 的 95% 分位（打乱时按定义为 5%）。cell_rows: 示例细胞的转录本；
    markers + nucleus（每个 MetaSpot 是否在核内）给出 b，缺则不画 b。"""
    spots = np.asarray(lens.get("null_spots", ()), np.int64)
    if not len(spots):
        return
    z = np.asarray(z)
    real = np.linalg.norm(z[spots], axis=1)
    null = np.linalg.norm(np.asarray(lens["null_z"], float), axis=1)
    threshold = float(np.quantile(null, 0.95))
    real_pct, null_pct = 100 * float(np.mean(real > threshold)), 100 * float(np.mean(null > threshold))
    stats = structure_direction_stats(z, lens, markers, nucleus) if markers and nucleus is not None else []
    with_cell = cell_rows is not None and spot_names is not None and len(cell_rows)
    widths = [0.7] + ([1.0] if stats else []) + ([1.15] if with_cell else [])
    fig, ax = _figure(
        14, 4.9, "Module 5 · Is the LENS signal real?  Same cells, gene positions shuffled", len(widths),
        subtitle="Control: in the same cells, every gene is moved to a random position inside its cell side (nucleus or cytoplasm); "
                 "MetaSpots keep their size and position.\nWhatever LENS still reports in shuffled cells is noise or comes from "
                 "MetaSpot size and position; labels use only the structure directions (b).",
        **({"width_ratios": widths} if len(widths) > 1 else {}))
    a = ax[0]
    bars = a.bar([0, 1], [real_pct, null_pct], width=0.6, color=[ACCENT, NULL])
    for bar, v in zip(bars, (real_pct, null_pct)):
        a.annotate(f"{v:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=9, fontweight="bold")
    a.set_xticks([0, 1], ["real", "shuffled"])
    a.set_ylim(0, max(real_pct, null_pct) * 1.3)
    a.grid(axis="y")
    a.set(ylabel="MetaSpots with a strong signal |z| (%)", title=f"{real_pct / max(null_pct, 1e-9):.1f}× more strong signal")
    if stats:
        b = ax[1]
        x = np.arange(len(stats))
        b.bar(x, [r for _, r, _ in stats], width=0.62, color=[_colour(n, i) for i, (n, _, _) in enumerate(stats)])
        for xi, (_, ratio, rho) in zip(x, stats):
            b.annotate(f"ρ {rho:.2f}", xy=(xi, ratio), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                       fontsize=8, color=INK2)
        b.axhline(1, color=INK2, lw=0.9)
        b.set_xticks(x, [n.replace(" ", "\n") for n, _, _ in stats])
        b.set_ylim(0, max(r for _, r, _ in stats) * 1.25)
        b.grid(axis="y")
        b.set(ylabel="structure-score spread, real ÷ shuffled\n(line at 1 = no more than shuffled)",
              xlabel="ρ above bars: same MetaSpot, real vs shuffled (low = little from size or position)",
              title="What labels use: score along each structure")
    if with_cell:
        c = ax[-1]
        idx = spot_names.get_indexer(cell_rows["metaspot"].astype(str))
        excess = np.linalg.norm(z[idx], axis=1) - pd.Series(null, index=spots).reindex(idx).to_numpy()
        lim = float(np.nanquantile(np.abs(excess), 0.98)) or 1.0
        x, y = cell_rows["x_location"].to_numpy(), cell_rows["y_location"].to_numpy()
        sc = c.scatter(x, y, s=3, c=excess, cmap=DIVERGING, vmin=-lim, vmax=lim, linewidths=0, rasterized=True)
        cell = str(cell_rows["cell_id"].iloc[0])
        for bounds, colour in ((cell_bounds, MUTED), (nucleus_bounds, INK2)):
            if bounds is not None:
                vtx = bounds[bounds["cell_id"].astype(str) == cell]
                if len(vtx):
                    c.plot(np.r_[vtx["vertex_x"], vtx["vertex_x"].iloc[:1]], np.r_[vtx["vertex_y"], vtx["vertex_y"].iloc[:1]],
                           color=colour, lw=0.9)
        c.set_aspect("equal")
        c.set_xticks([])
        c.set_yticks([])
        for spine in c.spines.values():
            spine.set_visible(False)
        c.set_title(f"Example cell ({len(cell_rows):,} transcripts): real − shuffled")
        cb = fig.colorbar(sc, ax=c, fraction=0.05, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("extra signal from genes (real − shuffled |z|)", color=INK2)
    _save(fig, out_dir, "05_lens_signal")


# ---------------------------------------------------------------- 06 MetaDomain
OBSERVED_MIN_COUNT = 200  # 只排名总条数 ≥ 200 的基因，免得罕见、几乎不表达的基因排到前面


def observed_domain_genes(counts, domain: np.ndarray, nucleus: Optional[np.ndarray], genes,
                          n_top: int = 6, min_count: int = OBSERVED_MIN_COUNT) -> dict:
    """每个 MetaDomain 实际观测富集最高的基因：{k: [(基因, 观测/期望)]}。

    counts: MetaSpot × 基因计数（adata1.X）；domain: 每个 MetaSpot 的 MetaDomain（-1 = 背景）；
    nucleus: 每个 MetaSpot 是否在核内（None = 不分侧）。期望 = 同侧背景的基因占比 × 该 MetaDomain 在这一侧的总条数，
    两侧相加；只排名 MetaDomain + 背景总数 ≥ min_count 的基因。"""
    domain = np.asarray(domain, np.int64)
    side = np.zeros(len(domain), np.int64) if nucleus is None else np.asarray(nucleus, bool).astype(np.int64)
    ids = np.unique(domain[domain >= 0])
    n_groups = len(ids) + 1
    group = side * n_groups + np.searchsorted(ids, domain) + (domain >= 0)  # 背景为 0，MetaDomain 为 1..K
    agg = sp.csr_matrix((np.ones(len(domain), np.float32), (group, np.arange(len(domain)))), shape=(2 * n_groups, len(domain)))
    table = np.asarray((agg @ sp.csr_matrix(counts)).todense(), float).reshape(2, n_groups, -1)  # 侧 × 组 × 基因
    background = table[:, 0, :]
    genes = np.asarray(genes).astype(str)
    out = {}
    for j, k in enumerate(ids, start=1):
        observed = table[:, j, :]
        expected = (background / np.maximum(background.sum(1, keepdims=True), 1) * observed.sum(1, keepdims=True)).sum(0)
        observed = observed.sum(0)
        ratio = (observed + 1) / (expected + 1)
        keep = observed + background.sum(0) >= min_count
        order = np.argsort(-np.where(keep, ratio, -np.inf))[:n_top]
        out[int(k)] = [(genes[g], float(ratio[g])) for g in order if keep[g]]
    return out


@_styled
def plot_metadomains(out_dir: str, disc: Mapping, observed: Optional[Mapping[int, list]] = None) -> None:
    """每个 MetaDomain 的 MetaSpot 数、核内比例、实际观测富集最高的基因；真实 vs 打乱细胞超过阈值的比例。

    observed: observed_domain_genes 的结果；None 时退回 uns 里解码器方向的头部基因并在标题注明。"""
    ids = np.asarray(disc.get("metadomain", ()))
    if not len(ids):
        fig, ax = _figure(8, 2, "Module 5 · MetaDomains")
        ax[0].axis("off")
        ax[0].text(0.5, 0.5, "No MetaDomain beat the gene-shuffled cells", ha="center", va="center", color=INK2)
        _save(fig, out_dir, "06_metadomains")
        return
    text = lambda key: [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(disc[key])]
    side, top = text("side"), text("top_genes")
    metaspots = np.asarray(disc["metaspots"], float)
    nucleus = np.asarray(disc["nucleus_share"], float)
    fig, ax = _figure(13, 1.6 + 0.9 * max(len(ids), 2), "Module 5 · MetaDomains found from LENS without reference genes",
                      2, width_ratios=[2.3, 1])
    y = np.arange(len(ids))
    ax[0].barh(y, metaspots, height=0.6, color=ACCENT)
    ax[0].set_yticks(y, [f"MD{int(k)}\n{s} · {100 * f:.0f}% nuclear" for k, s, f in zip(ids, side, nucleus)])
    for yy, k, m, genes in zip(y, ids, metaspots, top):
        if observed is not None:
            label = "   ".join(f"{g} {r:.1f}×" for g, r in observed.get(int(k), []))
        else:
            label = ", ".join(genes.split(",")[:8])
        ax[0].text(m + metaspots.max() * 0.02, yy, label, va="center", fontsize=7.5, color=INK2)
    ax[0].set_xlim(0, metaspots.max() * 2.3)
    ax[0].invert_yaxis()
    ax[0].grid(axis="x")
    _count_axis(ax[0], "x")
    ax[0].set(xlabel="MetaSpots", title=(f"Size and most enriched observed genes (vs same-side background, ≥ {OBSERVED_MIN_COUNT} transcripts)"
                                         if observed is not None
                                         else "Size and top genes (decoder profile, not observed counts)"))
    ax[1].barh(y - 0.17, np.asarray(disc["real_above"], float) * 100, 0.32, color=ACCENT, label="real cells")
    ax[1].barh(y + 0.17, np.asarray(disc["null_above"], float) * 100, 0.32, color=NULL, label="genes shuffled")
    ax[1].set_yticks(y, [f"MD{int(k)}" for k in ids])
    ax[1].invert_yaxis()
    ax[1].grid(axis="x")
    ax[1].legend(loc="lower right")
    ax[1].set(xlabel="MetaSpots above threshold (%)", title="Real vs shuffled")
    _save(fig, out_dir, "06_metadomains")


# ---------------------------------------------------------------- 07 注释验证
@_styled
def plot_annotation(out_dir: str, report: Mapping) -> None:
    """各结构留出 marker 富集倍数（95% CI）、标签占比、LENS 基因载荷的留出 marker AUROC 与随机基因集对照。"""
    structures = report.get("structures", {}) or {}
    names = list(structures)
    # 最终是否通过以报告顶层为准（lens_count 规则还要过 c 的门控）；各结构自己的 supported 只是 a 这一项
    final = set(report.get("supported_structures", [n for n in names if structures[n].get("supported")]))
    fig, ax = _figure(14, 4, "Annotation · validation with held-out reference markers", 3)
    for i, name in enumerate(names):
        info = structures[name]
        lv = (info.get("levels") or {}).get(info.get("level")) or {}
        enr = lv.get("marker_enrichment")
        if enr is None:
            ax[0].text(1.0, i, "  no markers in this dataset", va="center", fontsize=8, color=MUTED)
            continue
        colour = _colour(name, i) if name in final else NULL
        ax[0].plot([lv.get("ci_low", enr), lv.get("ci_high", enr)], [i, i], color=colour, lw=2, solid_capstyle="round")
        ax[0].plot(enr, i, "o", ms=8, color=colour, mec="white", mew=1.2)
    ax[0].axvline(1, color=INK2, lw=0.8)
    ax[0].set_yticks(range(len(names)), [f"{n}{'' if n in final else '  (not labelled)'}" for n in names])
    ax[0].set_ylim(-0.6, len(names) - 0.4)
    ax[0].invert_yaxis()
    ax[0].grid(axis="x")
    ax[0].set(xlabel="held-out marker RNA, labelled regions / rest (95% CI)", title="Enrichment check")
    shares = report.get("label_metaspot_fraction") or {}
    labs = sorted(shares, key=lambda k: -shares[k])
    ax[1].barh(range(len(labs)), [100 * shares[k] for k in labs], height=0.65,
               color=[_colour(k, i) for i, k in enumerate(labs)])
    ax[1].set_yticks(range(len(labs)), labs)
    ax[1].invert_yaxis()
    ax[1].grid(axis="x")
    ax[1].set(xlabel="MetaSpots with this final label (%)", title="Final labels")
    lc = ((report.get("lens_count") or {}).get("structures") or {})
    rows = [(n, float(lc[n]["auroc"]), lc[n]["null_mean"], lc[n]["null_sd"], bool(lc[n].get("pass")))
            for n in names if n in lc and lc[n].get("auroc") is not None and np.isfinite(float(lc[n]["auroc"]))]
    _dot_vs_null(ax[2], rows, "held-out marker AUROC on LENS gene loadings")
    ax[2].axvline(0.5, color=INK2, lw=0.8)
    ax[2].set_title("Gate: above random gene sets")
    _save(fig, out_dir, "07_annotation_validation")


# ---------------------------------------------------------------- 08 深度偏差
@_styled
def plot_share_vs_depth(out_dir: str, table: pd.DataFrame) -> None:
    """细胞 × 结构表：每个结构的标签占比随细胞一侧 RNA 数的变化（按深度十分位）。"""
    t = table[table["kind"] == "structure"]
    fig, ax = _figure(11, 3.9, "Cell × structure table · labelled share against RNA depth (compare cells at matched depth)", 2)
    for a, (flag, side_name) in zip(ax, ((True, "Nucleus side"), (False, "Cytoplasm side"))):
        sub = t[t["in_nucleus"] == flag]
        ends = []
        for i, (name, rows) in enumerate(sub.groupby("feature", observed=True)):
            if rows["transcript_share"].fillna(0).eq(0).all():
                continue
            bin_ = pd.qcut(rows["n_transcripts"], 10, duplicates="drop")
            curve = rows.groupby(bin_, observed=True).agg(depth=("n_transcripts", "median"), share=("transcript_share", "mean"))
            colour = _colour(name, i)
            a.plot(curve["depth"], 100 * curve["share"], "-", color=colour, label=name)
            a.plot(curve["depth"], 100 * curve["share"], "o", ms=4, color=colour, mec="white", mew=0.8)
            ends.append((curve["depth"].iloc[-1], 100 * curve["share"].iloc[-1], name))
        a.set_xscale("log")
        a.margins(x=0.18)
        a.autoscale_view()  # 先定坐标范围，标签间距才按最终刻度换算
        _end_labels(a, ends)
        a.grid(axis="y")
        a.set(xlabel="transcripts on this side of the cell (decile medians)", ylabel="mean labelled share (%)",
              title=side_name)
        handles, labels = a.get_legend_handles_labels()
        if handles:
            a.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(handles))
    _save(fig, out_dir, "08_structure_share_vs_depth")


# ---------------------------------------------------------------- 09 示例细胞
def pick_example_cells(depth: pd.Series, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9, 0.97)) -> List[str]:
    """按 RNA 深度分位数选示例细胞（确定性，不看任何结果）。depth: cell_id → 转录本数；同数按 cell_id 排。"""
    depth = depth[depth > 0]
    depth.index = depth.index.astype(str)
    depth = depth.sort_index().sort_values(kind="stable")
    picks = [str(depth.index[int(q * (len(depth) - 1))]) for q in quantiles]
    return list(dict.fromkeys(picks))


@_styled
def plot_example_cells(out_dir: str, rows: pd.DataFrame, cell_bounds: Optional[pd.DataFrame],
                       nucleus_bounds: Optional[pd.DataFrame], cells: List[str]) -> None:
    """每个示例细胞一行：转录本 + 细胞/核轮廓、MetaSpot 划分、最终亚细胞标签、MetaDomain。

    rows: 这些细胞的转录本（cell_id, x_location, y_location, metaspot, subcellular_annotation[, meta-domain]）。"""
    columns: List[Tuple[str, Optional[str]]] = [("Transcripts", None), ("MetaSpots", "metaspot")]
    if "subcellular_annotation" in rows.columns:
        columns.append(("Subcellular label", "subcellular_annotation"))
    if "meta-domain" in rows.columns:
        columns.append(("MetaDomain", "meta-domain"))
    fig = Figure(figsize=(2.9 * len(columns), 2.9 * len(cells) + 1.0))
    axes = np.atleast_2d(fig.subplots(len(cells), len(columns)))
    _header(fig, "Example cells, chosen only by RNA depth (10–97% quantiles)",
            "Columns: raw transcripts (grey outline = cell, blue line = nucleus) → MetaSpots (one colour each) → "
            "final structure label → MetaDomain")
    rng = np.random.default_rng(0)
    seen_labels: dict = {}
    seen_domains: set = set()
    by_cell = {c: g for c, g in rows.groupby(rows["cell_id"].astype(str), observed=True)}

    def outline(ax, bounds, cell, colour, lw):
        if bounds is None:
            return
        v = bounds[bounds["cell_id"].astype(str) == cell]
        if len(v):
            ax.plot(np.r_[v["vertex_x"], v["vertex_x"].iloc[:1]], np.r_[v["vertex_y"], v["vertex_y"].iloc[:1]], color=colour, lw=lw)

    for r, cell in enumerate(cells):
        pts = by_cell.get(cell)
        if pts is None:
            continue
        x, y = pts["x_location"].to_numpy(), pts["y_location"].to_numpy()
        for c, (title, col) in enumerate(columns):
            ax = axes[r, c]
            if col is None:
                ax.scatter(x, y, s=1.2, c=INK2, linewidths=0, rasterized=True)
            elif col == "metaspot":
                codes = pd.factorize(pts[col])[0]
                palette = rng.random((codes.max() + 1, 3)) * 0.75 + 0.15
                ax.scatter(x, y, s=1.2, c=palette[codes], linewidths=0, rasterized=True)
            elif col == "meta-domain":
                md = pd.to_numeric(pts[col], errors="coerce").fillna(-1).astype(int).to_numpy()
                ax.scatter(x[md < 0], y[md < 0], s=1.2, c=STRUCTURE_COLOURS["Cytosol"], linewidths=0, rasterized=True)
                for k in np.unique(md[md >= 0]):
                    m = md == k
                    ax.scatter(x[m], y[m], s=4, marker=_DOMAIN_MARKERS[k % len(_DOMAIN_MARKERS)],
                               c=_DOMAIN_COLOURS[k % len(_DOMAIN_COLOURS)], linewidths=0, rasterized=True)
                    seen_domains.add(int(k))
            else:
                lab = pts[col].astype(str).to_numpy()
                for l in pd.unique(lab):
                    seen_labels.setdefault(l, _colour(l, len(seen_labels)))
                ax.scatter(x, y, s=1.2, c=[seen_labels[l] for l in lab], linewidths=0, rasterized=True)
            outline(ax, cell_bounds, cell, MUTED, 0.8)
            outline(ax, nucleus_bounds, cell, ACCENT if col is None else INK2, 0.9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, loc="center", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{cell}\n{len(pts):,} transcripts", fontsize=8, color=INK2)
    order = list(STRUCTURE_COLOURS)
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=seen_labels[k], label=k)
               for k in sorted(seen_labels, key=lambda k: order.index(k) if k in order else 99)]
    handles += [Line2D([], [], marker=_DOMAIN_MARKERS[k % len(_DOMAIN_MARKERS)], ls="", ms=6,
                       color=_DOMAIN_COLOURS[k % len(_DOMAIN_COLOURS)], label=f"MD{k}") for k in sorted(seen_domains)]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 11), fontsize=8.5, handletextpad=0.2,
                   columnspacing=1.2)
    _save(fig, out_dir, "09_example_cells", bottom=0.025)


# ---------------------------------------------------------------- 入口
def write_run_figures(out_dir: str, adata1, final_df: Optional[pd.DataFrame] = None,
                      cell_features: Optional[pd.DataFrame] = None,
                      cell_bounds: Optional[pd.DataFrame] = None,
                      nucleus_bounds: Optional[pd.DataFrame] = None,
                      example_cells: Optional[List[str]] = None,
                      markers: Optional[Mapping[str, List[str]]] = None) -> None:
    """运行结束时画 02、03、05–09（01 在预处理、04 在 LENS 训练后各自写出）。缺输入的图跳过。

    markers: 注释所用的参考 marker 集合（结构 → 基因名），03 图用；报告里只存了个数。
    example_cells: 已选好的示例细胞（回填时按全量深度预先选）；None 时按 final_df 的深度选。"""
    report = adata1.uns.get("evidence_annotation")
    report = json.loads(report) if isinstance(report, str) else (report or {})
    jobs: List[Tuple[str, Callable[[], None]]] = []
    if {"cell_id", "log_n"} <= set(adata1.obs.columns):
        jobs.append(("02_metaspots", lambda: plot_metaspots(out_dir, adata1.obs)))
    ge = adata1.uns.get("gene_embedding")
    if ge and markers:
        jobs.append(("03_gene_embedding", lambda: plot_gene_embedding(out_dir, ge, markers)))
    lens = adata1.uns.get("lens")
    if lens and "X_lens" in adata1.obsm:
        def lens_signal():
            cell = pick_lens_example_cell(adata1.obs, lens.get("null_spots", ()))
            rows = None
            if cell is not None and final_df is not None and {"metaspot", "x_location", "y_location"} <= set(final_df.columns):
                rows = final_df.loc[final_df["cell_id"].isin([cell]), ["cell_id", "x_location", "y_location", "metaspot"]]
            nucleus = ((adata1.obs["subcellular_broad_compartment"].astype(str) == "Nucleus").to_numpy()
                       if "subcellular_broad_compartment" in adata1.obs.columns else None)
            plot_lens_signal(out_dir, adata1.obsm["X_lens"], lens, adata1.obs_names.astype(str), rows, cell_bounds, nucleus_bounds,
                             markers, nucleus)
        jobs.append(("05_lens_signal", lens_signal))
    if adata1.uns.get("metadomain_discovery") is not None:
        def domains():
            observed = None
            if adata1.X is not None and "meta-domain" in adata1.obs.columns:
                obs = adata1.obs
                nucleus = ((obs["subcellular_broad_compartment"].astype(str) == "Nucleus").to_numpy()
                           if "subcellular_broad_compartment" in obs.columns else None)
                domain = pd.to_numeric(obs["meta-domain"].astype(str), errors="coerce").fillna(-1).astype(int).to_numpy()
                observed = observed_domain_genes(adata1.X, domain, nucleus, adata1.var_names)
            plot_metadomains(out_dir, adata1.uns["metadomain_discovery"], observed)
        jobs.append(("06_metadomains", domains))
    if report.get("structures"):
        jobs.append(("07_annotation_validation", lambda: plot_annotation(out_dir, report)))
    if cell_features is not None and len(cell_features):
        jobs.append(("08_structure_share_vs_depth", lambda: plot_share_vs_depth(out_dir, cell_features)))
    if final_df is not None and len(final_df):
        def examples():
            cells = example_cells or pick_example_cells(final_df["cell_id"].value_counts())
            keep = [c for c in ("cell_id", "x_location", "y_location", "metaspot", "subcellular_annotation", "meta-domain")
                    if c in final_df.columns]
            rows = final_df.loc[final_df["cell_id"].isin(cells), keep]
            plot_example_cells(out_dir, rows, cell_bounds, nucleus_bounds, cells)
        jobs.append(("09_example_cells", examples))
    for name, job in jobs:
        try:
            job()
        except Exception as exc:  # 诊断图失败不影响运行结果
            _LOG.warning("Diagnostic figure %s failed: %s", name, exc)
