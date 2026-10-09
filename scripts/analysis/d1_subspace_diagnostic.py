import argparse
import itertools
import json
import os
import re
import sys

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

TASKS = ["sentiment", "headline", "ner", "finer", "xbrl_extract",
         "xbrl_term", "formula", "financebench"]
POOL_SUFFIX = {"r8": "8bits_r8", "rslora": "8bits_r8_rslora", "dora": "8bits_r8_dora"}
PROJS = ["q_proj", "k_proj", "v_proj"]
KEY_RE = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.lora_(A|B)\.weight")

def load_factors(task, suffix):
    
    path = snapshot_download(f"wangd12/{task}_llama_3_1_8b_{suffix}")
    fn = os.path.join(path, "adapter_model.safetensors")
    out = {}
    with safe_open(fn, framework="pt") as f:
        for key in f.keys():
            m = KEY_RE.search(key)
            if not m:
                continue
            layer, proj, factor = int(m.group(1)), m.group(2), m.group(3)
            t = f.get_tensor(key).float().numpy().astype(np.float64)
            out.setdefault((layer, proj), {})[factor] = t
    return out

def orthobasis(mat):
    
    q, _ = np.linalg.qr(mat)
    return q

def mean_cos(basis_a, basis_b):
    
    s = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())

def chance_level(dim, r, n_draws=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = [mean_cos(orthobasis(rng.standard_normal((dim, r))),
                     orthobasis(rng.standard_normal((dim, r))))
            for _ in range(n_draws)]
    return float(np.mean(vals)), float(np.std(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="r8", choices=POOL_SUFFIX)
    ap.add_argument("--out", default=res("d1"))
    args = ap.parse_args()
    suffix = POOL_SUFFIX[args.pool]

    factors = {t: load_factors(t, suffix) for t in TASKS}
    cells = sorted(factors[TASKS[0]].keys())
    print(f"pool={args.pool}: {len(TASKS)} adapters, {len(cells)} (layer, proj) cells")

    bases = {
        t: {cell: {"A": orthobasis(fs[cell]["A"].T), "B": orthobasis(fs[cell]["B"])}
            for cell, fs in [(c, factors[t]) for c in cells]}
        for t in TASKS
    }

    rows = []
    for (t1, t2), (layer, proj) in itertools.product(itertools.combinations(TASKS, 2), cells):
        rows.append({
            "task_i": t1, "task_j": t2, "layer": layer, "proj": proj,
            "sim_A": mean_cos(bases[t1][(layer, proj)]["A"], bases[t2][(layer, proj)]["A"]),
            "sim_B": mean_cos(bases[t1][(layer, proj)]["B"], bases[t2][(layer, proj)]["B"]),
        })

    r = factors[TASKS[0]][cells[0]]["A"].shape[0]
    dims = {p: {"in": factors[TASKS[0]][(0, p)]["A"].shape[1],
                "out": factors[TASKS[0]][(0, p)]["B"].shape[0]} for p in PROJS}
    chance = {p: {"A": chance_level(dims[p]["in"], r), "B": chance_level(dims[p]["out"], r)}
              for p in PROJS}

    summary = {}
    for p in PROJS:
        pr = [x for x in rows if x["proj"] == p]
        sa, sb = np.array([x["sim_A"] for x in pr]), np.array([x["sim_B"] for x in pr])
        ca, cb = chance[p]["A"], chance[p]["B"]
        summary[p] = {
            "sim_A_mean": float(sa.mean()), "sim_A_std": float(sa.std()),
            "sim_B_mean": float(sb.mean()), "sim_B_std": float(sb.std()),
            "chance_A_mean": ca[0], "chance_A_std": ca[1],
            "chance_B_mean": cb[0], "chance_B_std": cb[1],
            "excess_A": float(sa.mean() - ca[0]), "excess_B": float(sb.mean() - cb[0]),
            "excess_A_sigma": float((sa.mean() - ca[0]) / ca[1]),
            "excess_B_sigma": float((sb.mean() - cb[0]) / cb[1]),
            "dims": dims[p],
        }

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, f"d1_{args.pool}.json")
    with open(out_json, "w") as f:
        json.dump({"pool": args.pool, "r": r, "summary": summary, "pairs": rows}, f, indent=1)

    print(f"\n{'proj':7s} {'sim_A':>7s} {'chanceA':>8s} {'excessA(sig)':>13s}   "
          f"{'sim_B':>7s} {'chanceB':>8s} {'excessB(sig)':>13s}")
    for p in PROJS:
        s = summary[p]
        print(f"{p:7s} {s['sim_A_mean']:7.4f} {s['chance_A_mean']:8.4f} "
              f"{s['excess_A']:+.4f} ({s['excess_A_sigma']:+6.1f}σ)   "
              f"{s['sim_B_mean']:7.4f} {s['chance_B_mean']:8.4f} "
              f"{s['excess_B']:+.4f} ({s['excess_B_sigma']:+6.1f}σ)")
    print(f"\nwrote {out_json}")

if __name__ == "__main__":
    main()
