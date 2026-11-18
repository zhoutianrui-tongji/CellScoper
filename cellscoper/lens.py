"""MetaSpot masked-transcript model (LENS): a neural MetaSpot representation of subcellular structure.

Each MetaSpot is a set of transcripts (gene token + position). A two-layer Transformer reads the visible
transcripts of the MetaSpot plus up to 32 transcripts of the same cell side around it (context), and pools
the MetaSpot's own tokens and the context by attention into a 32-d vector z. Training hides 25% of a
MetaSpot's transcripts (every copy of a hidden gene, also in the context) and predicts the hidden genes with

    log p(gene | MetaSpot) = log B[cell side, gene] + W[gene] . z + b[gene]   (softmax over the vocabulary)

B is a learnt mixture of the cell side's own composition (without this MetaSpot), a rank-20 NMF profile of
the cell side and the global composition, so z only explains where the MetaSpot deviates from its own cell:
structure, not cell type. Gene tokens are the frozen multiscale spatial gene embedding. 10% of cells are held
out for early stopping on the masked log-likelihood gain over B alone (nats per masked transcript), a
label-free score. z is centred per cell side, as the spatial gene embedding is.

Scale: counts are sparse and the low-rank profile is kept as NMF factors, so the GPU holds ~5 GB at full Atera
(~170k cells); per-transcript tables move to host memory above 4 GB, and spots_per_epoch keeps training time fixed.
"""
from __future__ import annotations

import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

from .spatial_gene_embedding import MULTISCALE_NN, fit_gene_embedding, median_nearest_neighbour_distance

LENS_MIN_VOCAB = 50  # genes with fewer transcripts share one input token and are never predicted
N_ENCODER_LAYERS = 2
MAX_LEN, D, Z_DIM, MASK_P, CONTEXT_NN, RANK = 32, 96, 32, 0.25, 14.0, 20
GPU_TABLE_MAX_BYTES = 4 * 2**30
NMF_FIT_MAX = 20000  # cell sides; more are projected onto factors fitted on a sample of this many
LENS_BACKGROUNDS = ("learned", "own")
# background="own": pseudo-counts of the cell side's low-rank profile added to its own counts, so genes the cell side has not
# shown elsewhere keep a small probability
OWN_BACKGROUND_PSEUDOCOUNT = 10.0


def _context(xyz, key, local, spot_key, centroid, radius):
    """MetaSpot x MAX_LEN context rows (-1 = empty): nearest transcripts of the same cell side outside the MetaSpot,
    within ``radius``."""
    S = len(centroid)
    ctx = np.full((S, MAX_LEN), -1, np.int64)
    order = np.argsort(key, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1, len(order)]
    spots_by_key = pd.Series(np.arange(S)).groupby(spot_key).indices
    for a, b in zip(bounds[:-1], bounds[1:]):
        members = order[a:b]
        qs = spots_by_key.get(key[members[0]])
        if qs is None:
            continue
        dist, nb = cKDTree(xyz[members]).query(centroid[qs], k=min(len(members), MAX_LEN + 32),
                                               distance_upper_bound=radius)
        dist, nb = np.atleast_2d(dist), np.atleast_2d(nb)
        if len(members) == 1:  # query with k=1 returns column vectors
            dist, nb = dist.reshape(-1, 1), nb.reshape(-1, 1)
        # 整批处理这一侧的 MetaSpot（与逐个循环结果相同）：按距离顺序保留半径内、不属于自身的转录本，取前 MAX_LEN 个
        valid = np.isfinite(dist)
        cand = members[np.where(valid, nb, 0)]
        keep = valid & (local[cand] != qs[:, None])
        slot = np.cumsum(keep, axis=1) - 1
        r, c = np.nonzero(keep & (slot < MAX_LEN))
        ctx[qs[r], slot[r, c]] = cand[r, c]
    return ctx


def _resolve_device(torch_module: Any, requested: Any = "auto") -> Any:
    """``auto``: the CUDA device with the most free memory (concurrent runs spread over GPUs), else CPU. ``cuda``
    without an available device raises instead of falling back silently."""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if normalized == "cpu" or (normalized == "auto" and not torch_module.cuda.is_available()):
        return torch_module.device("cpu")
    if not torch_module.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but this PyTorch build/runtime has no available CUDA device")
    cuda = torch_module.cuda
    return torch_module.device("cuda", max(range(cuda.device_count()), key=lambda i: cuda.mem_get_info(i)[0]))


def smooth_neighbours_of(centroid, spot_key, k):
    """每个细胞侧：(行号, 每行自身与最近 k - 1 个同侧 MetaSpot 的行号)。只依赖位置，多个分数可共用。"""
    groups = []
    for idx in pd.Series(np.arange(len(centroid))).groupby(spot_key).indices.values():
        kk = min(k, len(idx))
        j = np.asarray(cKDTree(centroid[idx]).query(centroid[idx], k=kk)[1]).reshape(len(idx), kk)
        groups.append((idx, idx[j]))
    return groups


def smooth_with(Z, groups):
    """在 smooth_neighbours_of 给出的近邻上取均值（与 _smooth 逐位相同）。"""
    out = Z.copy()
    for idx, members in groups:
        out[idx] = Z[members].mean(1)
    return out


def _smooth(Z, centroid, spot_key, k):
    """Mean z over each MetaSpot and its k - 1 nearest MetaSpots (centroids) of the same cell side."""
    return smooth_with(Z, smooth_neighbours_of(centroid, spot_key, k))


def lens_embedding(
    point_df: pd.DataFrame,
    metaspot_names,
    *,
    metaspot_col: str = "metaspot",
    max_epochs: int = 200,
    patience: int = 5,
    spots_per_epoch: int = 0,
    patience_units: int = 4_000_000,
    max_units: int = 100_000_000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    min_gene_count: int = 100,
    max_cells: int = 20000,
    seed: int = 42,
    device: str = "auto",
    log=None,
    n_jobs: int = 1,
    null_cells: int = 0,
    background: str = "learned",
    absolute_position: bool = True,
) -> Tuple[np.ndarray, dict]:
    """(MetaSpot x 32 LENS embedding aligned to ``metaspot_names``, centred per cell side; diagnostics).

    MetaSpots without transcripts get zero rows. ``spots_per_epoch`` > 0 trains each epoch on that many random
    MetaSpots and validates on a fixed subset, so training cost does not grow with the data; early stopping then
    counts MetaSpots seen (``patience_units`` without improvement, at most ``max_units``).

    Genes still visible in the input (MetaSpot + context) are removed from B: every copy of a hidden gene is hidden, so
    a visible gene is never the answer; without this, ruling out visible genes earns gain with no spatial information.

    ``null_cells`` > 0 also encodes up to that many random cells with gene identities shuffled within each cell side
    (positions kept), giving the null of any read-out through W: diagnostics ``null_spots`` (rows of ``metaspot_names``)
    and ``null_z``, processed as the real z. The real z is unaffected.

    ``background`` sets B: "learned" (the mixture above) or "own" = the cell side's own counts without this MetaSpot plus
    OWN_BACKGROUND_PSEUDOCOUNT pseudo-counts of its low-rank profile, with no global part. With "own", composition that
    differs between cells is already in B, so z and W only learn where a MetaSpot departs from its own cell; with genes
    shuffled within cells, W no longer ranks held-out reference genes (with "learned" it still does).

    ``absolute_position=False`` gives every transcript the same d_nuc_norm / d_cell_norm (.5): tokens keep the nucleus flag
    and their position relative to the MetaSpot, but not where in the cell they sit. With the own background the radial
    position otherwise dominates z in simulations; held-out labels on real data are unchanged.

    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if background not in LENS_BACKGROUNDS:
        raise ValueError(f"LENS background must be one of {LENS_BACKGROUNDS}, got {background!r}")
    say = log or (lambda *_: None)
    t0 = time.time()
    dev = _resolve_device(torch, device)
    amp = dev.type == "cuda"  # bf16 encoder: ~1.5x faster, same held-out gain
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    # ---------- data: transcripts ordered by MetaSpot ----------
    names = pd.Index(metaspot_names).astype(str)
    spot_all = pd.Categorical(point_df[metaspot_col].astype(str), categories=names).codes.astype(np.int64)
    rows = np.flatnonzero(spot_all >= 0)
    rows = rows[np.argsort(spot_all[rows], kind="stable")]
    spot_of = spot_all[rows]
    cells, cell_names = pd.factorize(point_df["cell_id"].astype(str).to_numpy()[rows])
    gene_codes, genes = pd.factorize(point_df["feature_name"].astype(str).to_numpy()[rows])
    axes = ["x_location", "y_location"]
    if "z_location" in point_df.columns and point_df["z_location"].notna().all():
        axes.append("z_location")
    xyz = np.zeros((len(rows), 3))
    xyz[:, :len(axes)] = point_df[axes].to_numpy(np.float64)[rows]
    side = point_df["in_nucleus"].to_numpy(bool)[rows]
    geo = [point_df[c].to_numpy(np.float32)[rows] if absolute_position and c in point_df.columns
           else np.full(len(rows), 0.5, np.float32) for c in ("d_nuc_norm", "d_cell_norm")]
    pos = np.nan_to_num(np.column_stack(geo + [side]).astype(np.float32), nan=0.5)
    n_pos = pos.shape[1] + 4  # + position relative to the MetaSpot centroid + context flag
    n_rows, n_pos_columns = len(rows), pos.shape[1]
    del spot_all, rows, geo  # 以下这些每条转录本一行的中间数组用完就释放：全量 Atera 每个都有几 GB

    counts = np.bincount(gene_codes, minlength=len(genes))
    vocab = np.flatnonzero(counts >= LENS_MIN_VOCAB)
    V = len(vocab)
    if V < 2:
        raise ValueError(f"LENS needs >= 2 genes with {LENS_MIN_VOCAB} transcripts, found {V}")
    tok_of = np.full(len(genes), V)
    tok_of[vocab] = np.arange(V)
    tok = tok_of[gene_codes]
    del tok_of, counts
    nn_dist = median_nearest_neighbour_distance(cells, xyz[:, :2], seed=seed)

    present, starts = np.unique(spot_of, return_index=True)  # MetaSpots with transcripts, contiguous blocks
    S = len(present)
    sizes = np.diff(np.r_[starts, n_rows])
    local = np.repeat(np.arange(S), sizes)
    centroid = np.add.reduceat(xyz, starts, axis=0) / sizes[:, None]
    spot_side = np.add.reduceat(side.astype(float), starts) / sizes > 0.5
    spot_cell = cells[starts]
    spot_key = spot_cell * 2 + spot_side.astype(int)
    del spot_of
    say(f"LENS: {n_rows} transcripts, {S} MetaSpots, vocabulary {V} genes, device {dev}")

    # MetaSpot x MAX_LEN transcript index; larger MetaSpots keep a random subset
    # MetaSpots with more than 32 transcripts (a few percent) are represented by a random subset of 32
    shuffled = np.lexsort((rng.random(n_rows), local))
    rank = np.arange(n_rows) - starts[local[shuffled]]
    keep = rank < MAX_LEN
    idx = np.full((S, MAX_LEN), -1, np.int64)
    idx[local[shuffled][keep], rank[keep]] = shuffled[keep]
    del shuffled, rank, keep

    # context: nearest transcripts of the same cell side around the centroid, outside the MetaSpot
    key = cells * 2 + side.astype(int)
    ctx = _context(xyz, key, local, spot_key, centroid, CONTEXT_NN * nn_dist)
    del local, side, starts, sizes
    scale = np.full(S, 10 * nn_dist, np.float32)  # length unit of the relative positions

    # background compositions per cell side: own counts, rank-20 NMF profile, global
    ok = tok < V
    # sparse cell side x gene counts and NMF factors (not the dense profile): full Atera (~340k cell sides x 15.6k
    # genes) would need ~21 GB per dense matrix on the GPU
    C = csr_matrix((np.ones(int(ok.sum()), np.float32), (key[ok], tok[ok])), shape=(len(cell_names) * 2, V))
    C.sum_duplicates()
    n_key = np.asarray(C.sum(1), np.float32).ravel()
    glob = np.asarray(C.sum(0)).ravel() / C.sum()
    del ok
    Fq = csr_matrix(C.multiply(1 / np.maximum(n_key, 1)[:, None]), dtype=np.float32)
    nmf = NMF(min(RANK, *Fq.shape), init="nndsvda", max_iter=300, random_state=0)
    if Fq.shape[0] <= NMF_FIT_MAX:
        Wn = nmf.fit_transform(Fq.toarray())
    else:  # fit on a sample of cell sides, then project all of them in chunks
        nmf.fit(Fq[np.sort(rng.choice(Fq.shape[0], NMF_FIT_MAX, replace=False))].toarray())
        Wn = np.vstack([nmf.transform(Fq[a:a + NMF_FIT_MAX].toarray()) for a in range(0, Fq.shape[0], NMF_FIT_MAX)])

    del Fq
    # frozen gene tokens: multiscale spatial gene embedding
    E0 = np.zeros((V + 1, D), np.float32)
    for i, m in enumerate(MULTISCALE_NN):
        kept, emb = fit_gene_embedding(cells, gene_codes, xyz[:, :len(axes)], len(genes), pair_radius=m * nn_dist,
                                       dim=D // len(MULTISCALE_NN), min_gene_count=min_gene_count, max_cells=max_cells,
                                       seed=0, n_jobs=n_jobs)
        block = np.zeros((len(genes), D // len(MULTISCALE_NN)), np.float32)
        block[kept, :emb.shape[1]] = emb
        block /= max(float(np.linalg.norm(block[kept], axis=1).mean()), 1e-12)
        E0[:V, i * block.shape[1]:(i + 1) * block.shape[1]] = block[vocab]
    del cells, gene_codes
    say(f"LENS inputs ready ({time.time() - t0:.0f}s)")

    class LENS(nn.Module):
        def __init__(self):
            super().__init__()
            self.gene = nn.Embedding.from_pretrained(torch.from_numpy(E0), freeze=True)
            self.pos = nn.Sequential(nn.Linear(n_pos, D), nn.GELU(), nn.Linear(D, D))
            layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, N_ENCODER_LAYERS, enable_nested_tensor=False)
            self.query = nn.Parameter(torch.zeros(D))
            self.attn = nn.Linear(D, D)
            self.to_z = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, Z_DIM))
            self.W = nn.Linear(Z_DIM, V)  # decoder: gene g's log-enrichment at a MetaSpot = W_g . z + b_g
            self.mix3 = nn.Parameter(torch.zeros(3))  # softmax weights of own, low-rank, global background

        def forward(self, g, p, visible):
            h = self.enc(self.gene(g) + self.pos(p), src_key_padding_mask=~visible)
            w = torch.softmax((self.attn(h) * self.query).sum(-1).masked_fill(~visible, -1e9), 1)
            return self.to_z((w.unsqueeze(-1) * h).sum(1))

    model = LENS().to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    # per-transcript and per-MetaSpot tables: on the GPU when small, else in host memory and each batch copies its rows
    # over (slower per epoch, but whole sections fit)
    table_bytes = idx.nbytes * 2 + n_rows * (8 + 4 * (3 + n_pos_columns)) + S * 24
    tab_dev = dev if table_bytes <= GPU_TABLE_MAX_BYTES else torch.device("cpu")
    H = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(tab_dev)
    idx_h = H(np.hstack([idx, ctx]))
    del idx, ctx
    feat_h = H(np.r_[np.column_stack([xyz, pos]), np.zeros((1, 3 + n_pos_columns))].astype(np.float32))  # xyz | pos
    del xyz, pos
    centroid_h = H(centroid.astype(np.float32))
    scale_h = H(scale)
    tok_h = H(np.r_[tok, V])  # index -1 -> rare token row, never visible
    key_h = H(spot_key)
    is_ctx_t = T(np.r_[np.zeros(MAX_LEN), np.ones(MAX_LEN)].astype(np.float32))
    crow_t, col_t, val_t = T(C.indptr.astype(np.int64)), T(C.indices.astype(np.int32)), T(C.data)  # full Atera ~325M nonzeros
    nkey_t, glob_t = T(n_key), T(glob.astype(np.float32))
    Wn_t, Hn_t = T(Wn.astype(np.float32)), T(nmf.components_.astype(np.float32))
    own_tok = is_ctx_t == 0

    nnz_h = H(np.diff(C.indptr))
    del C, centroid, scale
    say(f"LENS tables: {table_bytes / 2**30:.1f} GiB on {tab_dev}")

    def counts(keys, total):  # dense rows of the sparse cell side x gene counts; total = their nonzeros (no GPU sync)
        a, n = crow_t[keys], crow_t[keys + 1] - crow_t[keys]
        r = torch.repeat_interleave(torch.arange(len(keys), device=dev), n, output_size=total)
        at = torch.arange(total, device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n, output_size=total) + a[r]
        out = torch.zeros(len(keys), V, device=dev)
        out[r, col_t[at].long()] = val_t[at]
        return out

    def batch(spots, gen):
        sp = torch.from_numpy(spots).to(tab_dev)
        ii_h = idx_h[sp]
        valid = ii_h.to(dev) >= 0
        g = tok_h[ii_h].to(dev)
        f = feat_h[ii_h].to(dev)
        rel = ((f[..., :3] - centroid_h[sp].to(dev).unsqueeze(1)) / scale_h[sp].to(dev).view(-1, 1, 1)).clamp(-3, 3)
        p = torch.cat([f[..., 3:], rel, is_ctx_t.expand(len(spots), -1).unsqueeze(-1)], -1)
        eligible = valid & (g < V) & own_tok  # only the MetaSpot's own transcripts are targets
        if gen is None:
            return g, p, valid, None, None
        masked = eligible & (torch.rand(valid.shape, generator=gen, device=dev) < MASK_P)
        none = eligible.any(1) & ~masked.any(1)  # at least one target per MetaSpot
        if none.any():
            r = torch.rand(valid.shape, generator=gen, device=dev).masked_fill(~eligible, -1)
            masked[none, r[none].argmax(1)] = True
        hidden = valid & ((g.unsqueeze(2) == g.unsqueeze(1)) & masked.unsqueeze(1)).any(2) | masked
        own = torch.zeros(len(spots), V + 1, device=dev).scatter_add_(
            1, g.masked_fill(~(valid & own_tok), V), (valid & own_tok).float())[:, :V]
        keys_h = key_h[sp]
        keys = keys_h.to(dev)
        Cb = counts(keys, int(nnz_h[keys_h].sum())) - own
        prof = (Wn_t[keys] @ Hn_t).clamp(min=0)
        prof = prof / prof.sum(1, keepdim=True).clamp(min=1e-12)
        w3 = torch.softmax(model.mix3, 0)
        if background == "own":
            logB = torch.log((Cb + OWN_BACKGROUND_PSEUDOCOUNT * prof)
                             / ((nkey_t[keys] - own.sum(1)).clamp(min=0) + OWN_BACKGROUND_PSEUDOCOUNT).unsqueeze(1) + 1e-9)
        else:
            logB = torch.log(w3[0] * Cb / (nkey_t[keys] - own.sum(1)).clamp(min=1).unsqueeze(1) + w3[1] * prof
                             + w3[2] * glob_t + 1e-9)
        # 可见基因不会是答案：从背景里去掉（被遮住的基因所有拷贝都被遮住）
        vis = valid & ~hidden
        seen = torch.zeros(len(spots), V + 1, device=dev).scatter_(1, g.masked_fill(~vis, V), 1.0)[:, :V] > 0
        logB = logB.masked_fill(seen, -1e4)
        return g, p, valid & ~hidden, eligible & hidden, logB

    def loss(spots, gen, with_gain, per_spot=False):
        g, p, visible, target, logB = batch(spots, gen)
        keep = visible.any(1) & target.any(1)
        g, p, visible, target, logB = g[keep], p[keep], visible[keep], target[keep], logB[keep]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            z = model(g, p, visible)
        full = F.log_softmax(logB + model.W(z.float()), 1)
        r, c = torch.nonzero(target, as_tuple=True)
        tg = g[r, c]
        nll = -full[r, tg].mean()
        if not with_gain:
            return nll, None, len(tg)
        d = full[r, tg] - F.log_softmax(logB, 1)[r, tg]
        if per_spot:  # (MetaSpot index, gain per masked transcript) for the MetaSpots that had targets
            return T(spots)[keep][r], d
        return nll, d.mean(), len(tg)

    hold = np.random.default_rng(0).permutation(len(cell_names)) < 0.1 * len(cell_names)
    val_spots = np.flatnonzero(hold[spot_cell])
    train_spots = np.flatnonzero(~hold[spot_cell])
    if spots_per_epoch:
        val_spots = np.sort(np.random.default_rng(1).choice(
            val_spots, min(len(val_spots), max(100_000, int(spots_per_epoch) // 5)), replace=False))
        # stopping counted in MetaSpots seen: a fixed number of small sampled epochs stops too early
        patience = max(int(patience), int(np.ceil(patience_units / int(spots_per_epoch))))
        max_epochs = max(int(max_epochs), int(np.ceil(max_units / int(spots_per_epoch))))
    if not len(val_spots) or not len(train_spots):
        raise ValueError("LENS needs at least 10 cells (10% held out for early stopping)")

    def evaluate():
        model.eval()
        gen = torch.Generator(device=dev).manual_seed(123)
        n_tot, g_tot = 0, 0.0
        with torch.no_grad():
            for b in range(0, len(val_spots), 4096):
                _, gain, n = loss(val_spots[b:b + 4096], gen, True)
                n_tot += n
                g_tot += float(gain) * n
        model.train()
        return g_tot / max(n_tot, 1)

    history, best, best_state, stale, best_epoch = [], -np.inf, None, 0, 0
    gen = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(1, int(max_epochs) + 1):
        run = []
        perm = (rng.choice(train_spots, min(len(train_spots), int(spots_per_epoch)), replace=False)
                if spots_per_epoch else rng.permutation(train_spots))
        for b in range(0, len(perm), batch_size):
            nll, _, _ = loss(perm[b:b + batch_size], gen, False)
            opt.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(nll))
        gain = evaluate()
        history.append({"epoch": ep, "train_nll": float(np.mean(run)), "heldout_gain": gain,
                        "seconds": time.time() - t0})
        say(f"LENS epoch {ep}: train NLL {history[-1]['train_nll']:.4f}, held-out gain {gain:.4f} nats ({time.time() - t0:.0f}s)")
        if gain > best + 1e-4:
            best, stale, best_epoch = gain, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    # held-out gain per MetaSpot (one masking draw): how many MetaSpots carry composition beyond their cell side
    spot_gain = np.full(S, np.nan, np.float32)
    with torch.no_grad():
        gen_eval = torch.Generator(device=dev).manual_seed(123)
        for b in range(0, len(val_spots), 4096):
            sp, d = loss(val_spots[b:b + 4096], gen_eval, True, per_spot=True)
            sp, d = sp.cpu().numpy(), d.float().cpu().numpy()
            spot_gain[np.unique(sp)] = (np.bincount(sp, weights=d, minlength=S) / np.maximum(np.bincount(sp, minlength=S), 1))[np.unique(sp)]

    def encode(spot_rows):
        out_z = np.zeros((len(spot_rows), Z_DIM), np.float32)
        with torch.no_grad():
            for b in range(0, len(spot_rows), 8192):
                spots = spot_rows[b:b + 8192]
                g, p, visible, _, _ = batch(spots, None)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    out_z[b:b + len(spots)] = model(g, p, visible).float().cpu().numpy()
        out_z -= pd.DataFrame(out_z).groupby(spot_key[spot_rows]).transform("mean").to_numpy(np.float32)  # remove each cell side's mean
        return out_z

    Z = encode(np.arange(S))
    null_spots, null_z = np.zeros(0, np.int64), np.zeros((0, Z_DIM), np.float32)
    if null_cells:
        r_null = np.random.default_rng(seed + 1)
        pick = r_null.choice(len(cell_names), min(len(cell_names), int(null_cells)), replace=False)
        spot_rows = np.flatnonzero(np.isin(spot_cell, pick))
        shuffled = tok.copy()
        shuffled[np.lexsort((r_null.random(len(tok)), key))] = tok[np.argsort(key, kind="stable")]
        tok_h = H(np.r_[shuffled, V])  # batch() reads the table through this name
        null_spots, null_z = present[spot_rows], encode(spot_rows)
    out = np.zeros((len(names), Z_DIM), np.float32)
    out[present] = Z
    loadings = model.W.weight.detach().float().cpu().numpy()  # vocabulary x Z_DIM: gene g's log-enrichment at a MetaSpot = W_g . z
    w3 = torch.softmax(model.mix3, 0).detach().cpu().numpy()
    del model, opt, best_state, crow_t, col_t, val_t
    if dev.type == "cuda":  # later modules run in the same process; free the cached GPU memory for other jobs
        torch.cuda.empty_cache()
    heldout_spot_gain = np.full(len(names), np.nan, np.float32)
    heldout_spot_gain[present] = spot_gain
    diag = {"epochs": len(history), "best_heldout_gain": float(best), "history": history, "heldout_spot_gain": heldout_spot_gain,
            "best_epoch": best_epoch, "stop_reason": "early_stop" if stale >= patience else "max_epochs",
            "patience_epochs": int(patience), "max_epochs": int(max_epochs), "lr": float(lr), "batch_size": int(batch_size),
            "n_train_metaspots": int(len(train_spots)), "n_val_metaspots": int(len(val_spots)),
            "n_val_cells": int(hold.sum()),
            "background": background, "absolute_position": bool(absolute_position),
            "background_weights_own_lowrank_global": [float(x) for x in w3] if background == "learned" else [1.0, 0.0, 0.0],
            "vocabulary": int(V), "device": str(dev),
            "spots_per_epoch": int(spots_per_epoch),
            "gene_names": np.asarray(genes[vocab]).astype(str), "gene_loadings": loadings,
            "null_spots": null_spots, "null_z": null_z}
    say(f"LENS done: {len(history)} epochs, best held-out gain {best:.4f} nats/transcript ({time.time() - t0:.0f}s)")
    return out, diag
