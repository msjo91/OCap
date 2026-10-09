import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

from src.pool import TASKS, load_pool, write_merged_adapter
from src.merge import RULES, merged_rank
from src import asym

def recon_error(pool, merged, cell=(16, "q_proj")):
    
    exact = np.mean([(pool[t][cell]["B"] @ pool[t][cell]["A"]) for t in pool], axis=0)
    got = merged[cell]["B"] @ merged[cell]["A"]
    return float(np.linalg.norm(got - exact) / np.linalg.norm(exact))

def _unchanged(merged, out_dir):
    
    f = os.path.join(out_dir, "adapter_model.safetensors")
    if not os.path.exists(f):
        return False
    import torch
    from safetensors.torch import load_file
    from src.pool import KEY_RE
    try:
        old = load_file(f)
    except Exception:
        return False
    want = {f"base_model.model.model.layers.{l}.self_attn.{p}.lora_{fac}.weight"
            for (l, p) in merged for fac in ("A", "B")}
    if set(old) != want:
        return False
    for key, t in old.items():
        m = KEY_RE.search(key)
        cell = (int(m.group(1)), m.group(2))
        new = torch.tensor(np.ascontiguousarray(merged[cell][m.group(3)]), dtype=torch.bfloat16)
        if new.shape != t.shape or not torch.equal(new, t):
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--pool", default="r8", choices=["r8", "rslora"])
    ap.add_argument("--out", default=res("merges"))
    args = ap.parse_args()

    pool = load_pool(pool=args.pool)
    print(f"loaded pool ({args.pool}): {len(pool)} adapters")

    for rule in args.rules.split(","):
        fn, kwargs = RULES[rule]
        t0 = time.time()
        merged = fn(pool, **kwargs)
        for cell, fs in merged.items():
            assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), f"{rule}: non-finite factors at {cell}"
            assert np.abs(fs["B"]).max() > 0, f"{rule}: all-zero B at {cell}"
        err = recon_error(pool, merged) if rule in ("concat", "svd_prod8", "svd_prod64") else float("nan")
        rank = merged_rank(merged)
        out_dir = os.path.join(args.out, rule)
        if _unchanged(merged, out_dir):
            with open(os.path.join(out_dir, "MERGE_INFO.txt"), "w") as fh:
                fh.write(f"rule={rule} kwargs={kwargs} pool={args.pool} n={len(pool)} tasks={sorted(pool)}\n")
            print(f"{rule:12s} unchanged (identical factors already on disk) -> weights not rewritten", flush=True)
            continue
        write_merged_adapter(merged, out_dir, rank,
                             description=f"rule={rule} kwargs={kwargs} pool={args.pool} "
                                         f"n={len(pool)} tasks={sorted(pool)}")
        print(f"{rule:12s} rank={rank:3d}  rel_err_vs_uniform_mean={err:.4f}  "
              f"{time.time()-t0:.1f}s -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
