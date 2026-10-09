import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import ROOT, res

from src import lmm_pool as LP
from src.merge import RULES, merged_rank
from src import asym

DEFAULT_RULES = "concat,ties64,dare64,trim_g50,isocts_g50,isocts_neq"

def recon_error(pool, merged, cell=(16, "self_attn.q_proj")):
    
    exact = np.mean([pool[t][cell]["B"] @ pool[t][cell]["A"] for t in pool], axis=0)
    got = merged[cell]["B"] @ merged[cell]["A"]
    return float(np.linalg.norm(got - exact) / np.linalg.norm(exact))

def build(rule, pool, projector, out_root, pool_desc):
    fn, kwargs = RULES[rule]
    n_r = len(pool) * next(iter(next(iter(pool.values())).values()))["A"].shape[0]
    if "rank" in kwargs and kwargs["rank"] < n_r:
        kwargs = {**kwargs, "rank": n_r}
    if rule == "isocts":
        kwargs = {**kwargs, "s": None}
    t0 = time.time()
    merged = fn(pool, **kwargs)
    for cell, fs in merged.items():
        assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), f"{rule}: non-finite factors at {cell}"
        assert np.abs(fs["B"]).max() > 0, f"{rule}: all-zero B at {cell}"
    assert set(merged) == set(next(iter(pool.values()))), f"{rule}: cell set changed"
    rank = merged_rank(merged)
    err = recon_error(pool, merged) if rule == "concat" else float("nan")
    t_merge = time.time() - t0
    out_dir = os.path.join(out_root, rule)
    LP.write_llava_merged(merged, out_dir, rank, projector=projector,
                          description=f"rule={rule} kwargs={kwargs} {pool_desc} projector=mean")
    print(f"{rule:12s} rank={rank:4d} cells={len(merged)} recon_err={err:.2e} "
          f"merge={t_merge:.1f}s total={time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--root", default=os.path.join(ROOT, "external", "mm_mergebench"))
    ap.add_argument("--out", default=res("lmm_merges"))
    ap.add_argument("--tasks", default=",".join(LP.TASKS))
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    t0 = time.time()
    pool, projs = LP.load_llava_pool(args.root, tasks)
    n_cells = len(next(iter(pool.values())))
    print(f"loaded pool: {len(pool)} clients x {n_cells} cells in {time.time() - t0:.1f}s", flush=True)
    projector = LP.mean_projector(projs)
    pool_desc = f"pool=mm_mergebench n={len(pool)} tasks={sorted(pool)}"
    for rule in args.rules.split(","):
        build(rule, pool, projector, args.out, pool_desc)

if __name__ == "__main__":
    main()
