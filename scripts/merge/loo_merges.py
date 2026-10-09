import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src.pool import TASKS, load_pool, write_merged_adapter
from src.merge import RULES, merged_rank
from scripts.merge.make_merges import _unchanged


def build(pool, rule, out_dir, extra=""):
    fn, kw = RULES[rule]
    n_r = len(pool) * next(iter(next(iter(pool.values())).values()))["A"].shape[0]
    if kw.get("rank", 0) > n_r:
        kw = {**kw, "rank": n_r}
    if kw.get("k_common", 0) * 2 > n_r:
        kw = {**kw, "k_common": n_r // 2}
    if kw.get("k_c", 0) >= n_r:
        kw = {**kw, "k_c": n_r // 2}
    t0 = time.time()
    merged = fn(pool, **kw)
    for cell, fs in merged.items():
        assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), f"{out_dir}: non-finite at {cell}"
    rank = merged_rank(merged)
    desc = f"rule={rule} kwargs={kw} n={len(pool)} tasks={sorted(pool)} {extra}"
    if _unchanged(merged, out_dir):
        print(f"{os.path.basename(out_dir):28s} unchanged", flush=True); return
    write_merged_adapter(merged, out_dir, rank, description=desc)
    print(f"{os.path.basename(out_dir):28s} rank={rank:3d} {time.time()-t0:.0f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="concat,trim_g50,normeq_med")
    ap.add_argument("--skip_loo", action="store_true"); ap.add_argument("--skip_rslora", action="store_true")
    args = ap.parse_args()
    rules = args.rules.split(",")
    if not args.skip_loo:
        pool = load_pool(pool="r8")
        for c in TASKS:
            sub = {t: pool[t] for t in TASKS if t != c}
            for rule in rules:
                build(sub, rule, res("merges_loo", f"{rule}-no_{c}"), extra=f"left_out={c} pool=r8")
    if not args.skip_rslora:
        pool = load_pool(pool="rslora")
        for rule in rules:
            build(pool, rule, res("merges_rslora", rule), extra="pool=rslora")

if __name__ == "__main__":
    main()
