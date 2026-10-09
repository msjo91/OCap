import numpy as np
import torch

def _cells(pool):
    first = next(iter(pool.values()))
    return sorted(first.keys())

def _delta(fs, dtype=np.float32):
    return (fs["B"].astype(dtype) @ fs["A"].astype(dtype))

def _svd_factorize(delta, rank, seed=0):
    
    t = torch.from_numpy(np.ascontiguousarray(delta))
    torch.manual_seed(seed)
    U, S, V = torch.svd_lowrank(t, q=min(rank + 16, min(t.shape)), niter=4)
    U, S, V = U[:, :rank], S[:rank], V[:, :rank]
    B = (U * S.sqrt()).numpy()
    A = (S.sqrt()[:, None] * V.T).numpy()
    return A.astype(np.float64), B.astype(np.float64)

def merge_concat(pool):
    tasks = sorted(pool)
    n = len(tasks)
    out = {}
    for cell in _cells(pool):
        A = np.vstack([pool[t][cell]["A"] for t in tasks])
        B = np.hstack([pool[t][cell]["B"] / n for t in tasks])
        out[cell] = {"A": A, "B": B}
    return out

TRAIN_SIZES = {"sentiment": 72824, "headline": 82161, "ner": 13549, "finer": 10002,
               "xbrl_term": 5856, "formula": 800, "financebench": 86, "xbrl_extract": None}

def merge_concat_weighted(pool, sizes=None):
    
    import os, glob
    sizes = dict(sizes or TRAIN_SIZES)
    if sizes.get("xbrl_extract") is None:
        from .paths import ROOT
        d = os.path.join(ROOT, "external", "FinLoRA", "data", "train", "xbrl_csv")
        n = 0
        for f in glob.glob(os.path.join(d, "*_train.csv")):
            n += max(sum(1 for _ in open(f, errors="ignore")) - 1, 0)
        sizes["xbrl_extract"] = n
    tasks = sorted(pool)
    tot = float(sum(sizes[t] for t in tasks))
    out = {}
    for cell in _cells(pool):
        A = np.vstack([pool[t][cell]["A"] for t in tasks])
        B = np.hstack([pool[t][cell]["B"] * (sizes[t] / tot) for t in tasks])
        out[cell] = {"A": A, "B": B}
    return out

def merge_avg_factors(pool):
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        out[cell] = {
            "A": np.mean([pool[t][cell]["A"] for t in tasks], axis=0),
            "B": np.mean([pool[t][cell]["B"] for t in tasks], axis=0),
        }
    return out

def merge_avg_factors_normmatch(pool):
    
    tasks = sorted(pool)
    out = merge_avg_factors(pool)
    for cell in _cells(pool):
        target = np.linalg.norm(np.mean([_delta(pool[t][cell]) for t in tasks], axis=0))
        got = np.linalg.norm(_delta(out[cell]))
        out[cell] = {"A": out[cell]["A"], "B": out[cell]["B"] * (target / max(got, 1e-12))}
    return out

def merge_svd_prod(pool, rank=8):
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        delta = np.mean([_delta(pool[t][cell]) for t in tasks], axis=0)
        A, B = _svd_factorize(delta, rank)
        out[cell] = {"A": A, "B": B}
    return out

def merge_ties(pool, rank=64, keep=0.2, elect=True, disjoint=True, mag_pool=None):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        deltas = [_delta(pool[t][cell]) for t in tasks]
        mags = [_delta(mag_pool[t][cell]) for t in tasks] if mag_pool is not None else deltas
        merged = _ties_combine(deltas, mags, keep, elect, disjoint)
        A, B = _svd_factorize(merged, rank)
        out[cell] = {"A": A, "B": B}
    return out

def _ties_combine(deltas, mags, keep=0.2, elect=True, disjoint=True):
    
    trimmed, trimmed_m = [], []
    for d, m in zip(deltas, mags):
        if keep < 1.0:
            thresh = np.quantile(np.abs(d), 1 - keep)
            mask = np.abs(d) >= thresh
            trimmed.append(np.where(mask, d, 0.0)); trimmed_m.append(np.where(mask, m, 0.0))
        else:
            trimmed.append(d); trimmed_m.append(m)
    stack = np.stack(trimmed); stack_m = np.stack(trimmed_m)
    if elect:
        sign = np.sign(stack.sum(axis=0))
        agree = np.where(np.sign(stack) == sign[None], stack_m, 0.0)
    else:
        agree = stack_m
    cnt = (agree != 0).sum(axis=0)
    if disjoint:
        return agree.sum(axis=0) / np.maximum(cnt, 1)
    mean_cnt = max(float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 1.0, 1.0)
    return agree.sum(axis=0) / mean_cnt

def merge_dare(pool, rank=64, p=0.9, seed=0):
    rng = np.random.default_rng(seed)
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        kept = []
        for t in tasks:
            d = _delta(pool[t][cell])
            mask = rng.random(d.shape) >= p
            kept.append(d * mask / (1 - p))
        merged = np.mean(kept, axis=0)
        A, B = _svd_factorize(merged, rank)
        out[cell] = {"A": A, "B": B}
    return out

def merge_proj_shared(pool, factor="B", k=16):
    
    tasks = sorted(pool)
    n = len(tasks)
    out = {}
    for cell in _cells(pool):
        if factor == "B":
            stacked = np.hstack([pool[t][cell]["B"] for t in tasks])
        else:
            stacked = np.hstack([pool[t][cell]["A"].T for t in tasks])
        Uf, _, _ = np.linalg.svd(stacked.astype(np.float32), full_matrices=False)
        U = Uf[:, :k]
        mean_delta = np.mean([_delta(pool[t][cell]) for t in tasks], axis=0)
        if factor == "B":
            proj = U @ (U.T @ mean_delta)
        else:
            proj = (mean_delta @ U) @ U.T
        A, B = _svd_factorize(proj, k)
        out[cell] = {"A": A, "B": B}
    return out

def merge_slao(pool, rank=8, order=None):
    
    tasks = order or sorted(pool)
    out = {}
    for cell in _cells(pool):
        A = pool[tasks[0]][cell]["A"].copy()
        B = pool[tasks[0]][cell]["B"].copy()
        for i, t in enumerate(tasks[1:], start=2):
            lam = 1.0 / np.sqrt(i)
            A = pool[t][cell]["A"].copy()
            B = B + lam * (pool[t][cell]["B"] - B)
        out[cell] = {"A": A, "B": B}
    return out

def merge_pico(pool, rank=64):
    
    tasks = sorted(pool)
    T_n = len(tasks)
    out = {}
    for cell in _cells(pool):
        stacked = np.hstack([pool[t][cell]["B"] for t in tasks]).astype(np.float32)
        U, sv, _ = np.linalg.svd(stacked, full_matrices=False)
        s = sv ** 2 / (sv ** 2).sum()
        alpha = 1.0 / (1.0 + (T_n - 1) * s)
        S = np.eye(U.shape[0], dtype=np.float32) + U @ np.diag(alpha - 1.0) @ U.T
        merged = np.mean([(S @ pool[t][cell]["B"].astype(np.float32))
                          @ pool[t][cell]["A"].astype(np.float32) for t in tasks], axis=0)
        A, B = _svd_factorize(merged, rank)
        out[cell] = {"A": A, "B": B}
    return out

def client_norms(pool, cell):
    
    return {t: float(np.linalg.norm(_delta(pool[t][cell]))) for t in pool}

def normalize_pool(pool, target="median", gamma=1.0, max_up=None):
    
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        v = np.array(list(norms.values()))
        if target == "median":
            tgt = float(np.median(v))
        elif target == "geomean":
            tgt = float(np.exp(np.mean(np.log(np.maximum(v, 1e-12)))))
        elif target == "mean":
            tgt = float(np.mean(v))
        else:
            raise ValueError(target)
        for t in pool:
            sc = (tgt / max(norms[t], 1e-12)) ** gamma
            if max_up is not None:
                sc = min(sc, max_up)
            out[t][cell] = {"A": pool[t][cell]["A"], "B": pool[t][cell]["B"] * sc}
    return out

def whiten_pool(pool, target="median"):
    
    if target not in ("median", "mean"):
        raise ValueError(target)
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        v = np.array(list(norms.values()))
        tgt = float(np.median(v)) if target == "median" else float(np.mean(v))
        for t in pool:
            B, A = pool[t][cell]["B"], pool[t][cell]["A"]
            r = A.shape[0]
            U, S, V = _lowrank_svd(B, A)
            k = len(S)
            A_out = np.zeros((r, A.shape[1])); B_out = np.zeros((B.shape[0], r))
            if k:
                scale = tgt / np.sqrt(k)
                A_out[:k] = V.T; B_out[:, :k] = U * scale
            out[t][cell] = {"A": A_out, "B": B_out}
    return out

def norm_match(merged, ref):
    
    out = {}
    for cell in merged:
        target = np.linalg.norm(_delta(ref[cell]))
        got = np.linalg.norm(_delta(merged[cell]))
        out[cell] = {"A": merged[cell]["A"], "B": merged[cell]["B"] * (target / max(got, 1e-12))}
    return out

def mad_tau(norms, k=3.0):
    
    n = np.asarray(list(norms), dtype=float)
    med = float(np.median(n))
    mad = float(np.median(np.abs(n - med)))
    return (med + k * mad) / max(med, 1e-12)

def cap_ratio_pool(pool, tau=3.0, k=3.0):
    
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        tau_c = mad_tau(norms.values(), k) if tau == "mad" else tau
        ceil = tau_c * float(np.median(list(norms.values())))
        for t in pool:
            sc = min(1.0, ceil / max(norms[t], 1e-12))
            out[t][cell] = {"A": pool[t][cell]["A"], "B": pool[t][cell]["B"] * sc}
    return out

def auto_tau(pool, cell, grid=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 1e9), eps=1e-12):
    
    best, best_cv = None, np.inf
    for tau in grid:
        capped = cap_ratio_pool({t: {cell: pool[t][cell]} for t in pool}, tau)
        B = np.hstack([capped[t][cell]["B"] for t in sorted(pool)]).astype(np.float64)
        A = np.vstack([capped[t][cell]["A"] for t in sorted(pool)]).astype(np.float64)
        _, S, _ = _lowrank_svd(B, A)
        if S.size == 0:
            continue
        cv = float(S.std() / max(S.mean(), eps))
        if cv < best_cv:
            best, best_cv = tau, cv
    return best, best_cv

def merge_adaptive(pool, tau=3.0, smoothing="mean", base="isocts", k=3.0, **kw):
    
    p = dcmerge_smooth_pool(pool, smoothing)
    if tau == "auto":
        out = {}
        for cell in _cells(p):
            t_star, _ = auto_tau(p, cell)
            one = cap_ratio_pool({k: {cell: p[k][cell]} for k in p}, t_star)
            merged = _BASE_RULES[base]({k: {cell: one[k][cell]} for k in one}, **kw)
            out[cell] = merged[cell]
        return out
    p = cap_ratio_pool(p, tau, k)
    return _BASE_RULES[base](p, **kw)

def merge_capped_raw(pool, tau="mad", base="isocts", **kw):
    
    return _BASE_RULES[base](cap_ratio_pool(pool, tau), **kw)

def merge_scaled(pool, rule="concat", lam=1.0):
    
    fn, kw = RULES[rule]
    merged = fn(pool, **kw)
    return {cell: {"A": fs["A"], "B": fs["B"] * lam} for cell, fs in merged.items()}

def merge_norm_matched(pool, rule="isocts_neq", ref="normeq_med", rank=64):
    
    fn, kw = RULES[rule]
    rf, rkw = RULES[ref]
    if "rank" in kw:
        kw = {**kw, "rank": rank}
    return norm_match(fn(pool, **kw), rf(pool, **rkw))

def merge_whiten(pool, target="median", base="isocts", **kw):
    
    return _BASE_RULES[base](whiten_pool(pool, target), **kw)

def cap_share_pool(pool, share=0.5, tol=1e-6):
    
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        v = np.array([norms[t] for t in pool])
        total = v.sum()
        k = int((v > 0).sum())
        if total <= 0 or v.max() / total <= share + tol:
            c = float(v.max())
        elif share < 1.0 / k - tol:
            c = float(v[v > 0].min())
        else:
            lo, hi = 0.0, float(v.max())
            for _ in range(100):
                c = 0.5 * (lo + hi)
                if c / np.minimum(v, c).sum() <= share:
                    lo = c
                else:
                    hi = c
            c = lo
        for t in pool:
            sc = min(1.0, c / max(norms[t], 1e-12))
            out[t][cell] = {"A": pool[t][cell]["A"], "B": pool[t][cell]["B"] * sc}
    return out

def merge_cap(pool, share=0.5, base="concat", **kw):
    
    return _BASE_RULES[base](cap_share_pool(pool, share), **kw)

def clip_cells_pool(pool, c=10.0):
    
    out = {t: {} for t in pool}
    for t in pool:
        norms = {cell: float(np.linalg.norm(_delta(pool[t][cell]))) for cell in pool[t]}
        ceil = c * float(np.median(list(norms.values())))
        for cell in pool[t]:
            sc = min(1.0, ceil / max(norms[cell], 1e-12))
            out[t][cell] = {"A": pool[t][cell]["A"], "B": pool[t][cell]["B"] * sc}
    return out

def merge_clipcell(pool, c=10.0, base="concat", **kw):
    return _BASE_RULES[base](clip_cells_pool(pool, c), **kw)

def merge_clip_normeq(pool, c=10.0, target="median", base="ties", gamma=0.5, max_up=1.0, **kw):
    
    return merge_normeq(clip_cells_pool(pool, c), target=target, base=base, gamma=gamma, max_up=max_up, **kw)

_BASE_RULES = {
    "concat": merge_concat, "svd_prod": merge_svd_prod, "ties": merge_ties,
    "dare": merge_dare, "pico": merge_pico, "proj_shared": merge_proj_shared,
}

def merge_normeq(pool, target="median", base="concat", gamma=1.0, max_up=None, **kw):
    
    return _BASE_RULES[base](normalize_pool(pool, target, gamma=gamma, max_up=max_up), **kw)

def merge_ties_split(pool, gamma_vote=1.0, gamma_mag=0.5, target="median", rank=64, keep=0.2):
    
    Pv = normalize_pool(pool, target, gamma=gamma_vote, max_up=None)
    Pm = normalize_pool(pool, target, gamma=gamma_mag, max_up=1.0)
    return merge_ties(Pv, rank=rank, keep=keep, elect=True, disjoint=True, mag_pool=Pm)

def _lowrank_svd(B, A, tol=1e-6):
    
    B = np.asarray(B, dtype=np.float64); A = np.asarray(A, dtype=np.float64)
    Qb, Rb = np.linalg.qr(B)
    Qa, Ra = np.linalg.qr(A.T)
    P, s, Wt = np.linalg.svd(Rb @ Ra.T)
    keep = s > tol * s[0] if s.size and s[0] > 0 else np.zeros(s.shape, bool)
    return Qb @ P[:, keep], s[keep], Qa @ Wt[keep].T

def _orth(M):
    
    P, _, Qt = np.linalg.svd(M, full_matrices=False)
    return P @ Qt

def _factors_from_svd(U, s, V, rank):
    
    order = np.argsort(-s)
    U, s, V = U[:, order], s[order], V[:, order]
    k = min(len(s), rank)
    root = np.sqrt(s[:k])
    B = np.zeros((U.shape[0], rank)); A = np.zeros((rank, V.shape[0]))
    B[:, :k] = U[:, :k] * root
    A[:k] = root[:, None] * V[:, :k].T
    return A, B

def merge_tsvm(pool, rank=64, alpha=1.0):
    
    tasks = sorted(pool)
    n = len(tasks)
    out = {}
    for cell in _cells(pool):
        Us, Ss, Vs = [], [], []
        for t in tasks:
            U, s, V = _lowrank_svd(pool[t][cell]["B"], pool[t][cell]["A"])
            k = min(len(s), min(U.shape[0], V.shape[0]) // n)
            Us.append(U[:, :k]); Ss.append(s[:k]); Vs.append(V[:, :k])
        U, S, V = np.hstack(Us), np.concatenate(Ss), np.hstack(Vs)
        A, B = _factors_from_svd(_orth(U), alpha * S, _orth(V), rank)
        out[cell] = {"A": A, "B": B}
    return out

def merge_isoc(pool, rank=64, alpha=1.0):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        U, s, V = _lowrank_svd(np.hstack([pool[t][cell]["B"] for t in tasks]),
                               np.vstack([pool[t][cell]["A"] for t in tasks]))
        flat = np.full(len(s), alpha * s.mean()) if len(s) else s
        A, B = _factors_from_svd(U, flat, V, rank)
        out[cell] = {"A": A, "B": B}
    return out

def merge_isocts(pool, rank=64, k_common=32, s=None, alpha=1.0):
    
    tasks = sorted(pool)
    n = len(tasks)
    s_ts = (rank - k_common) // n if s is None else s
    out = {}
    for cell in _cells(pool):
        U, sv, V = _lowrank_svd(np.hstack([pool[t][cell]["B"] for t in tasks]),
                                np.vstack([pool[t][cell]["A"] for t in tasks]))
        kc = min(k_common, len(sv))
        Uc = U[:, :kc]
        Us, Ss, Vs = [Uc], [sv[:kc]], [V[:, :kc]]
        for t in tasks:
            Bt, At = pool[t][cell]["B"], pool[t][cell]["A"]
            Bres = Bt - Uc @ (Uc.T @ Bt)
            Ut, st, Vt = _lowrank_svd(Bres, At)
            m = min(s_ts, len(st))
            Us.append(Ut[:, :m]); Ss.append(st[:m]); Vs.append(Vt[:, :m])
        Ustar, Sstar, Vstar = np.hstack(Us), np.concatenate(Ss), np.hstack(Vs)
        flat = np.full(len(Sstar), alpha * Sstar.mean()) if len(Sstar) else Sstar
        A, B = _factors_from_svd(_orth(Ustar), flat, _orth(Vstar), rank)
        out[cell] = {"A": A, "B": B}
    return out

def _knots_align(pool, cell, tasks):
    
    Qb, Rb = np.linalg.qr(np.hstack([np.asarray(pool[t][cell]["B"], dtype=np.float64) for t in tasks]))
    QRa = [np.linalg.qr(np.asarray(pool[t][cell]["A"], dtype=np.float64).T) for t in tasks]
    offsets = np.cumsum([0] + [q.shape[1] for q, _ in QRa])
    core = np.hstack([Rb[:, offsets[i]:offsets[i + 1]] @ Ra.T for i, (_, Ra) in enumerate(QRa)])
    P, s, Wt = np.linalg.svd(core)
    keep = s > 1e-6 * s[0] if s.size and s[0] > 0 else np.zeros(s.shape, bool)
    U = Qb @ P[:, keep]
    Zs = [s[keep, None] * (Wt[keep][:, offsets[i]:offsets[i + 1]] @ Qa.T) for i, (Qa, _) in enumerate(QRa)]
    return U, Zs

def merge_knots(pool, rank=64, method="ties", keep=0.2, elect=True, disjoint=True):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        U, Zs = _knots_align(pool, cell, tasks)
        if method == "ta":
            Zm = np.mean(Zs, axis=0)
        elif method == "ties":
            Zm = _ties_combine(Zs, Zs, keep=keep, elect=elect, disjoint=disjoint)
        else:
            raise ValueError(method)
        Uo, so, Vo = _lowrank_svd(U, Zm)
        A, B = _factors_from_svd(Uo, so, Vo, rank)
        out[cell] = {"A": A, "B": B}
    return out

def _dcmerge_smooth(s, smoothing="mean", rho=5.0):
    
    s = np.asarray(s, dtype=np.float64)
    if smoothing == "none" or s.size == 0:
        return s
    if smoothing == "mean":
        w = np.full(s.size, 1.0 / s.size)
    elif smoothing == "linear":
        ratio = min(float(rho), s[0] / s[-1]) if s[-1] > 0 else float(rho)
        w = np.linspace(ratio, 1.0, s.size)
        w = w / w.sum()
    else:
        raise ValueError(smoothing)
    return s.sum() * w

def dcmerge_smooth_pool(pool, smoothing="mean", rho=5.0):
    
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        for t in pool:
            B, A = pool[t][cell]["B"], pool[t][cell]["A"]
            r = A.shape[0]
            U, s, V = _lowrank_svd(B, A)
            k = len(s)
            A_out = np.zeros((r, A.shape[1])); B_out = np.zeros((B.shape[0], r))
            if k:
                A_out[:k] = V.T; B_out[:, :k] = U * _dcmerge_smooth(s, smoothing, rho)
            out[t][cell] = {"A": A_out, "B": B_out}
    return out

def _dcmerge_trim(mats, percent=1e-3):
    
    out = []
    for m in mats:
        flat = np.abs(m).ravel()
        k = max(1, int(percent * flat.size))
        if k >= flat.size:
            out.append(m); continue
        thr = np.partition(flat, flat.size - k)[flat.size - k]
        out.append(np.where(np.abs(m) >= thr, m, 0.0))
    return out

def merge_dcmerge(pool, rank=64, smoothing="mean", rho=5.0, trim_percent=1e-3,
                  aggregate="ties", mask=True, alpha=1.0):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        Us, Ss, Vs = [], [], []
        for t in tasks:
            U, s, V = _lowrank_svd(pool[t][cell]["B"], pool[t][cell]["A"])
            Us.append(U); Ss.append(_dcmerge_smooth(s, smoothing, rho)); Vs.append(V)
        off = np.cumsum([0] + [u.shape[1] for u in Us])
        K = int(off[-1])
        if K == 0:
            out[cell] = {"A": np.zeros((rank, pool[tasks[0]][cell]["A"].shape[1])),
                         "B": np.zeros((pool[tasks[0]][cell]["B"].shape[0], rank))}
            continue
        Ut = _orth(np.hstack(Us))
        Vt = _orth(np.hstack(Vs))
        Ms = [((Ut.T @ Us[i]) * Ss[i]) @ (Vs[i].T @ Vt) for i in range(len(tasks))]
        Ms = _dcmerge_trim(Ms, trim_percent)
        if aggregate == "ties":
            Mt = _ties_combine(Ms, Ms, keep=1.0, elect=True, disjoint=True)
        elif aggregate == "ta":
            Mt = np.mean(Ms, axis=0)
        else:
            raise ValueError(aggregate)
        if mask:
            blk = np.zeros((K, K))
            for i in range(len(tasks)):
                blk[off[i]:off[i + 1], off[i]:off[i + 1]] = 1.0
            Mt = Mt * blk
        U, s, V = _lowrank_svd(Ut, (alpha * Mt) @ Vt.T)
        A, B = _factors_from_svd(U, s, V, rank)
        out[cell] = {"A": A, "B": B}
    return out

def _orthobasis(mat):
    q, _ = np.linalg.qr(mat)
    return q

def _mean_cos(basis_a, basis_b):
    s = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())

def frame_similarity(pool, factor="A"):
    
    names = sorted(pool)
    cells = _cells(pool)
    bases = {t: {c: _orthobasis(pool[t][c]["A"].T if factor == "A" else pool[t][c]["B"])
                 for c in cells} for t in names}
    S = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = np.mean([_mean_cos(bases[names[i]][c], bases[names[j]][c]) for c in cells])
            S[i, j] = S[j, i] = v
    return names, S

def chance_frame_similarity(dim, r, n_draws=200, seed=0):
    
    rng = np.random.default_rng(seed)
    vals = [_mean_cos(_orthobasis(rng.standard_normal((dim, r))), _orthobasis(rng.standard_normal((dim, r))))
            for _ in range(n_draws)]
    return float(np.mean(vals)), float(np.std(vals))

def frame_clusters(names, S, tau=0.5):
    
    n = len(names)
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            k = stack.pop(); comp.append(k)
            for j in range(n):
                if j not in seen and S[k, j] > tau:
                    seen.add(j); stack.append(j)
        out.append(sorted(names[k] for k in comp))
    return sorted(out, key=lambda c: c[0])

def regime_info(pool, tau=0.5):
    
    names, S = frame_similarity(pool)
    cell = _cells(pool)[0]
    r, d_in = pool[names[0]][cell]["A"].shape
    ch_mean, ch_sd = chance_frame_similarity(d_in, r)
    off = [S[i, j] for i in range(len(names)) for j in range(i + 1, len(names))]
    above = [v for v in off if v > tau]; below = [v for v in off if v <= tau]
    return {"names": names, "sim": S.tolist(), "tau": tau,
            "chance_mean": ch_mean, "chance_sd": ch_sd,
            "clusters": frame_clusters(names, S, tau),
            "min_above": (min(above) if above else None), "max_below": (max(below) if below else None)}

def merge_regime_switch(pool, tau=0.5, product="trim_g50"):
    
    names, S = frame_similarity(pool)
    clusters = frame_clusters(names, S, tau)
    virtual = {"+".join(cl): merge_avg_factors({t: pool[t] for t in cl}) for cl in clusters}
    if len(virtual) == 1:
        return next(iter(virtual.values()))
    fn, kw = RULES[product]
    return fn(virtual, **kw)

_BASE_RULES.update({"tsvm": merge_tsvm, "isoc": merge_isoc, "isocts": merge_isocts,
                    "knots": merge_knots, "dcmerge": merge_dcmerge})

def _do_orthogonalize(mats, rows, steps=100, lr_scale=0.01):
    ws = [torch.tensor(m, dtype=torch.float64) for m in mats]
    ds = [torch.zeros_like(w, requires_grad=True) for w in ws]
    rms = float(torch.sqrt(torch.mean(torch.cat([w.flatten() for w in ws]) ** 2)))
    opt = torch.optim.Adam(ds, lr=lr_scale * max(rms, 1e-12))
    for _ in range(steps):
        opt.zero_grad()
        vs = [w + d for w, d in zip(ws, ds)]
        V = torch.cat(vs, 0) if rows else torch.cat(vs, 1)
        G = V @ V.T if rows else V.T @ V
        lo = (G ** 2).sum() - sum(((v @ v.T if rows else v.T @ v) ** 2).sum() for v in vs)
        lr_ = sum((d ** 2).sum() for d in ds)
        (lo + lr_).backward()
        opt.step()
    return [(w + d).detach().numpy() for w, d in zip(ws, ds)]

def merge_domerging(pool, lam=None, orthogonalize=True, steps=100, **kw):
    tasks = sorted(pool)
    n = len(tasks)
    lam = 1.0 / n ** 2 if lam is None else (1.0 / n if lam == "1/n" else lam)
    out = {}
    for cell in _cells(pool):
        As = [pool[t][cell]["A"].astype(np.float64) for t in tasks]
        Bs = [pool[t][cell]["B"].astype(np.float64) for t in tasks]
        if orthogonalize:
            As = _do_orthogonalize(As, rows=True, steps=steps)
            Bs = _do_orthogonalize(Bs, rows=False, steps=steps)
        alphas = []
        for A, B in zip(As, Bs):
            G = B.T @ B
            alphas.append(np.sqrt(np.maximum(np.einsum("ij,ik,kj->j", A, G, A), 1e-30)))
        asum = np.sum(alphas, axis=0)
        A_out = np.vstack([A * (asum / a)[None, :] for A, a in zip(As, alphas)]) * lam
        B_out = np.hstack(Bs)
        out[cell] = {"A": A_out.astype(np.float32), "B": B_out.astype(np.float32)}
    return out

RULES = {
    "concat": (merge_concat, {}),
    "concat_weighted": (merge_concat_weighted, {}),
    "avg_factors": (merge_avg_factors, {}),
    "avg_factors_nm": (merge_avg_factors_normmatch, {}),
    "svd_prod8": (merge_svd_prod, {"rank": 8}),
    "svd_prod64": (merge_svd_prod, {"rank": 64}),
    "ties64": (merge_ties, {"rank": 64, "keep": 0.2}),
    "dare64": (merge_dare, {"rank": 64, "p": 0.9}),
    "proj_b16": (merge_proj_shared, {"factor": "B", "k": 16}),
    "proj_b32": (merge_proj_shared, {"factor": "B", "k": 32}),
    "proj_a16": (merge_proj_shared, {"factor": "A", "k": 16}),
    "slao": (merge_slao, {"rank": 8}),
    "pico64": (merge_pico, {"rank": 64}),
    "normeq_med": (merge_normeq, {"target": "median", "base": "concat"}),
    "normeq_geo": (merge_normeq, {"target": "geomean", "base": "concat"}),
    "normeq_mean": (merge_normeq, {"target": "mean", "base": "concat"}),
    "normeq_pico64": (merge_normeq, {"target": "median", "base": "pico", "rank": 64}),
    "ties64_neq": (merge_normeq, {"target": "median", "base": "ties", "rank": 64, "keep": 0.2}),
    "dare64_neq": (merge_normeq, {"target": "median", "base": "dare", "rank": 64, "p": 0.9}),
    "normeq_g25": (merge_normeq, {"target": "median", "base": "concat", "gamma": 0.25, "max_up": 1.0}),
    "normeq_g50": (merge_normeq, {"target": "median", "base": "concat", "gamma": 0.5, "max_up": 1.0}),
    "normeq_g75": (merge_normeq, {"target": "median", "base": "concat", "gamma": 0.75, "max_up": 1.0}),
    "clipcell10": (merge_clipcell, {"c": 10.0, "base": "concat"}),
    "clipcell3": (merge_clipcell, {"c": 3.0, "base": "concat"}),
    "ties64_clip10": (merge_clipcell, {"c": 10.0, "base": "ties", "rank": 64, "keep": 0.2}),
    "trim_g50":   (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.5, "max_up": 1.0, "rank": 64, "keep": 0.2, "elect": False, "disjoint": False}),
    "elect_g50":  (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.5, "max_up": 1.0, "rank": 64, "keep": 1.0, "elect": True, "disjoint": True}),
    "tiesmean_g50": (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.5, "max_up": 1.0, "rank": 64, "keep": 0.2, "elect": True, "disjoint": False}),
    "ties64_g40": (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.4, "max_up": 1.0, "rank": 64, "keep": 0.2}),
    "ties64_g60": (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.6, "max_up": 1.0, "rank": 64, "keep": 0.2}),
    "ties_split": (merge_ties_split, {"gamma_vote": 1.0, "gamma_mag": 0.5, "rank": 64, "keep": 0.2}),
    "ties128_g50": (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.5, "max_up": 1.0, "rank": 128, "keep": 0.2}),
    "ties64_g50": (merge_normeq, {"target": "median", "base": "ties", "gamma": 0.5, "max_up": 1.0, "rank": 64, "keep": 0.2}),
    "trim_g50_clip10": (merge_clip_normeq, {"c": 10.0, "target": "median", "base": "ties", "gamma": 0.5, "max_up": 1.0, "rank": 64, "keep": 0.2, "elect": False, "disjoint": False}),
    "tsvm_g50":   (merge_normeq, {"target": "median", "base": "tsvm", "gamma": 0.5, "max_up": 1.0, "rank": 64}),
    "isoc_g50":   (merge_normeq, {"target": "median", "base": "isoc", "gamma": 0.5, "max_up": 1.0, "rank": 64}),
    "isocts_g50": (merge_normeq, {"target": "median", "base": "isocts", "gamma": 0.5, "max_up": 1.0, "rank": 64}),
    "knots_ties_g50": (merge_normeq, {"target": "median", "base": "knots", "gamma": 0.5, "max_up": 1.0, "rank": 64, "method": "ties", "keep": 0.2}),
    "adaptive":      (merge_adaptive, {"tau": 3.0, "base": "isocts", "rank": 64}),
    "adaptive_t10":  (merge_adaptive, {"tau": 10.0, "base": "isocts", "rank": 64}),
    "adaptive_t1":   (merge_adaptive, {"tau": 1.0, "base": "isocts", "rank": 64}),
    "adaptive_mad":  (merge_adaptive, {"tau": "mad", "base": "isocts", "rank": 64}),
    "domerging": (merge_domerging, {}),
    "domerging_l1n": (merge_domerging, {"lam": "1/n"}),
    "adaptive_mad_k2": (merge_adaptive, {"tau": "mad", "base": "isocts", "rank": 64, "k": 2.0}),
    "adaptive_mad_k4": (merge_adaptive, {"tau": "mad", "base": "isocts", "rank": 64, "k": 4.0}),
    "adaptive_tinf": (merge_adaptive, {"tau": 1e9, "base": "isocts", "rank": 64}),
    "capmad_isocts": (merge_capped_raw, {"tau": "mad", "base": "isocts", "rank": 64}),
    "capmad_concat": (merge_capped_raw, {"tau": "mad", "base": "concat"}),
    "capmad_dcmerge": (merge_capped_raw, {"tau": "mad", "base": "dcmerge", "rank": 64}),
    "capmed_concat": (merge_capped_raw, {"tau": 1.0, "base": "concat"}),
    "concat_nm_capmad": (merge_norm_matched, {"rule": "concat", "ref": "capmad_concat"}),
    "capmad_isoc":  (merge_capped_raw, {"tau": "mad", "base": "isoc", "rank": 64}),
    "capmad_tsvm":  (merge_capped_raw, {"tau": "mad", "base": "tsvm", "rank": 64}),
    "capmad_knots": (merge_capped_raw, {"tau": "mad", "base": "knots", "rank": 64, "method": "ties", "keep": 0.2}),
    "capmad_ties":  (merge_capped_raw, {"tau": "mad", "base": "ties", "rank": 64, "keep": 0.2}),
    "adaptive_mad_isoc":    (merge_adaptive, {"tau": "mad", "base": "isoc", "rank": 64}),
    "adaptive_mad_tsvm":    (merge_adaptive, {"tau": "mad", "base": "tsvm", "rank": 64}),
    "adaptive_mad_dcmerge": (merge_adaptive, {"tau": "mad", "base": "dcmerge", "rank": 64}),
    "adaptive_mad_concat": (merge_adaptive, {"tau": "mad", "base": "concat"}),
    "adaptive_mad_knots":   (merge_adaptive, {"tau": "mad", "base": "knots", "rank": 64, "method": "ties", "keep": 0.2}),
    "adaptive_mad_ties":    (merge_adaptive, {"tau": "mad", "base": "ties", "rank": 64, "keep": 0.2}),
    "concat_l2":  (merge_scaled, {"rule": "concat", "lam": 2.0}),
    "concat_l4":  (merge_scaled, {"rule": "concat", "lam": 4.0}),
    "concat_l6":  (merge_scaled, {"rule": "concat", "lam": 6.0}),
    "concat_l8":  (merge_scaled, {"rule": "concat", "lam": 8.0}),
    "concat_l05": (merge_scaled, {"rule": "concat", "lam": 0.5}),
    "isocts_white_s3": (merge_whiten, {"target": "median", "base": "isocts", "rank": 24, "k_common": 12}),
    "isocts_neq_s3":   (merge_normeq, {"target": "median", "base": "isocts", "gamma": 1.0, "max_up": None, "rank": 24, "k_common": 12}),
    "isoc_white_s3":   (merge_whiten, {"target": "median", "base": "isoc", "rank": 24}),
    "dcmerge_neq":   (merge_normeq, {"target": "median", "base": "dcmerge", "gamma": 1.0, "max_up": None, "rank": 64}),
    "dcmerge_g50":   (merge_normeq, {"target": "median", "base": "dcmerge", "gamma": 0.5, "max_up": 1.0, "rank": 64}),
    "dcmerge_trim4": (merge_dcmerge, {"rank": 64, "trim_percent": 4e-3}),
    "isocts_neq_nm": (merge_norm_matched, {"rule": "isocts_neq", "ref": "normeq_med", "rank": 64}),
    "isocts_g50_nm": (merge_norm_matched, {"rule": "isocts_g50", "ref": "normeq_g50"}),
    "isoc_white_nm": (merge_norm_matched, {"rule": "isoc_white", "ref": "concat_white"}),
    "isocts_white": (merge_whiten, {"target": "median", "base": "isocts", "rank": 64}),
    "isoc_white":   (merge_whiten, {"target": "median", "base": "isoc", "rank": 64}),
    "concat_white": (merge_whiten, {"target": "median", "base": "concat"}),
    "concat_cap50": (merge_cap, {"share": 0.5, "base": "concat"}),
    "trim_cap50":   (merge_cap, {"share": 0.5, "base": "ties", "rank": 64, "keep": 0.2, "elect": False, "disjoint": False}),
    "isocts_cap50": (merge_cap, {"share": 0.5, "base": "isocts", "rank": 64}),
    "isocts_cap33": (merge_cap, {"share": 1/3, "base": "isocts", "rank": 64}),
    "isocts_cap25": (merge_cap, {"share": 0.25, "base": "isocts", "rank": 64}),
    "isocts_g40":  (merge_normeq, {"target": "median", "base": "isocts", "gamma": 0.4, "max_up": 1.0, "rank": 64}),
    "isocts_g60":  (merge_normeq, {"target": "median", "base": "isocts", "gamma": 0.6, "max_up": 1.0, "rank": 64}),
    "isocts_g55":  (merge_normeq, {"target": "median", "base": "isocts", "gamma": 0.55, "max_up": 1.0, "rank": 64}),
    "isocts_g70":  (merge_normeq, {"target": "median", "base": "isocts", "gamma": 0.7, "max_up": 1.0, "rank": 64}),
    "isocts_neq":  (merge_normeq, {"target": "median", "base": "isocts", "gamma": 1.0, "max_up": None, "rank": 64}),
    "isocts_g100": (merge_normeq, {"target": "median", "base": "isocts", "gamma": 1.0, "max_up": 1.0, "rank": 64}),
    "isocts_white_p2": (merge_whiten, {"target": "median", "base": "isocts", "rank": 16, "k_common": 8}),
    "isocts_white_p3": (merge_whiten, {"target": "median", "base": "isocts", "rank": 24, "k_common": 12}),
    "isocts_white_p4": (merge_whiten, {"target": "median", "base": "isocts", "rank": 32, "k_common": 16}),
    "isocts_white_p6": (merge_whiten, {"target": "median", "base": "isocts", "rank": 48, "k_common": 24}),
    "dcmerge_p2": (merge_dcmerge, {"rank": 16, "smoothing": "mean", "trim_percent": 1e-3, "aggregate": "ties", "mask": True, "alpha": 1.0}),
    "dcmerge_p3": (merge_dcmerge, {"rank": 24, "smoothing": "mean", "trim_percent": 1e-3, "aggregate": "ties", "mask": True, "alpha": 1.0}),
    "dcmerge_p4": (merge_dcmerge, {"rank": 32, "smoothing": "mean", "trim_percent": 1e-3, "aggregate": "ties", "mask": True, "alpha": 1.0}),
    "dcmerge_p6": (merge_dcmerge, {"rank": 48, "smoothing": "mean", "trim_percent": 1e-3, "aggregate": "ties", "mask": True, "alpha": 1.0}),
    "knots_white": (merge_whiten, {"target": "median", "base": "knots", "rank": 64, "method": "ties", "keep": 0.2}),
    "ties_white":  (merge_whiten, {"target": "median", "base": "ties",  "rank": 64, "keep": 0.2}),
    "tsvm_white":  (merge_whiten, {"target": "median", "base": "tsvm",  "rank": 64}),
    "dcmerge_white": (merge_whiten, {"target": "median", "base": "dcmerge", "rank": 64}),
    "isocts_white_kc16": (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "k_common": 16}),
    "isocts_white_kc24": (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "k_common": 24}),
    "isocts_white_kc40": (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "k_common": 40}),
    "isocts_white_kc48": (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "k_common": 48}),
    "isocts_white_a05":  (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "alpha": 0.5}),
    "isocts_white_a075": (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "alpha": 0.75}),
    "isocts_white_a15":  (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "alpha": 1.5}),
    "isocts_white_a20":  (merge_whiten, {"target": "median", "base": "isocts", "rank": 64, "alpha": 2.0}),
    "isocts_white_mean": (merge_whiten, {"target": "mean",   "base": "isocts", "rank": 64}),
    "regime_switch": (merge_regime_switch, {"tau": 0.5, "product": "trim_g50"}),
    "regime_switch_t98": (merge_regime_switch, {"tau": 0.98, "product": "trim_g50"}),
    "tsvm": (merge_tsvm, {"rank": 64, "alpha": 1.0}),
    "isoc": (merge_isoc, {"rank": 64, "alpha": 1.0}),
    "isocts": (merge_isocts, {"rank": 64, "k_common": 32, "s": 4, "alpha": 1.0}),
    "knots_ties": (merge_knots, {"rank": 64, "method": "ties", "keep": 0.2}),
    "knots_ta": (merge_knots, {"rank": 64, "method": "ta"}),
    "dcmerge": (merge_dcmerge, {"rank": 64, "smoothing": "mean", "trim_percent": 1e-3,
                                "aggregate": "ties", "mask": True, "alpha": 1.0}),
    "dcmerge_linear": (merge_dcmerge, {"rank": 64, "smoothing": "linear", "rho": 5.0,
                                       "trim_percent": 1e-3, "aggregate": "ties",
                                       "mask": True, "alpha": 1.0}),
    "dcmerge_nosmooth": (merge_dcmerge, {"rank": 64, "smoothing": "none", "trim_percent": 1e-3,
                                         "aggregate": "ties", "mask": True, "alpha": 1.0}),
}

def rank_of(rule_name, n_clients, r=8):
    if rule_name == "concat":
        return n_clients * r
    if rule_name in ("avg_factors", "avg_factors_nm"):
        return r
    if rule_name.startswith("regime_switch"):
        raise ValueError("regime_switch rank depends on the detected clusters; use merged_rank(merged)")
    kw = RULES[rule_name][1]
    if "rule" in kw:
        return rank_of(kw["rule"], n_clients, r)
    if kw.get("base") == "concat":
        return n_clients * r
    return kw.get("rank", kw.get("k", r))

def merged_rank(merged):
    
    ranks = {fs["A"].shape[0] for fs in merged.values()}
    assert len(ranks) == 1, f"inconsistent ranks across cells: {ranks}"
    return ranks.pop()
