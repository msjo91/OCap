import numpy as np

from .merge import (RULES, _cells, _factors_from_svd, _lowrank_svd, normalize_pool, whiten_pool)

def shared_output_basis(pool, cell, k_c):
    
    stacked = np.hstack([pool[t][cell]["B"] for t in sorted(pool)]).astype(np.float64)
    U, _, _ = np.linalg.svd(stacked, full_matrices=False)
    if k_c >= stacked.shape[1]:
        raise ValueError(f"k_c={k_c} >= n*r={stacked.shape[1]}: the shared basis would span the whole "
                         "column space and the asymmetric projection would be a no-op")
    return U[:, :k_c]

def merge_asym_cts(pool, rank=64, k_c=32, beta=1.0, client_step="polar", isotropic=True, scale="sum"):
    
    if client_step == "polar":
        pool = whiten_pool(pool, target="median")
    elif client_step == "equalize":
        pool = normalize_pool(pool, "median", gamma=1.0, max_up=None)
    elif client_step != "none":
        raise ValueError(client_step)
    if scale not in ("sum", "mean"):
        raise ValueError(scale)
    tasks = sorted(pool)
    n = len(tasks)
    out = {}
    for cell in _cells(pool):
        U_c = shared_output_basis(pool, cell, k_c)
        Bs, As = [], []
        for t in tasks:
            B = pool[t][cell]["B"].astype(np.float64)
            B_t = beta * B + (1.0 - beta) * (U_c @ (U_c.T @ B))
            Bs.append(B_t / n if scale == "mean" else B_t)
            As.append(pool[t][cell]["A"].astype(np.float64))
        B_cat = np.hstack(Bs)
        A_cat = np.vstack(As)
        U, S, V = _lowrank_svd(B_cat, A_cat)
        if isotropic and S.size:
            S = np.full_like(S, S.mean())
        A, B = _factors_from_svd(U, S, V, rank)
        out[cell] = {"A": A, "B": B}
    return out

RULES.update({
    "asym_cts":         (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 0.5, "client_step": "polar"}),
    "asym_cts_eq":      (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 0.5, "client_step": "equalize"}),
    "asym_cts_raw":     (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 0.5, "client_step": "none"}),
    "asym_shared_only": (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 0.0, "client_step": "polar"}),
    "asym_full":        (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 1.0, "client_step": "polar"}),
    "asym_cts_k16":     (merge_asym_cts, {"rank": 64, "k_c": 16, "beta": 0.5, "client_step": "polar"}),
    "asym_cts_k48":     (merge_asym_cts, {"rank": 64, "k_c": 48, "beta": 0.5, "client_step": "polar"}),
    "asym_cts_k56":     (merge_asym_cts, {"rank": 64, "k_c": 56, "beta": 0.5, "client_step": "polar"}),
    "asym_cts_b25_k56": (merge_asym_cts, {"rank": 64, "k_c": 56, "beta": 0.25, "client_step": "polar"}),
    "asym_cts_b75_k56": (merge_asym_cts, {"rank": 64, "k_c": 56, "beta": 0.75, "client_step": "polar"}),
    "asym_cts_aniso":   (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 0.5, "client_step": "polar", "isotropic": False}),
    "asym_cts_mean":    (merge_asym_cts, {"rank": 64, "k_c": 32, "beta": 0.5, "client_step": "polar", "scale": "mean"}),
})
