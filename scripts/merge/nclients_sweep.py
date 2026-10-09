import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src.pool import TASKS, load_pool, write_merged_adapter
from src.merge import RULES, merged_rank, client_norms, _cells
from src import asym
from src import tasks as T
from scripts.merge.make_merges import _unchanged

OUT = res("merges_nclients")
LIGHT_FIRST = ["financebench", "formula", "xbrl_extract", "xbrl_term", "finer", "ner", "headline", "sentiment"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,4,6,8")
    ap.add_argument("--order", default="light_first", choices=["light_first", "heavy_first"])
    ap.add_argument("--rules", default="concat,isocts_white,dcmerge")
    args = ap.parse_args()
    order = LIGHT_FIRST if args.order == "light_first" else LIGHT_FIRST[::-1]

    pool = load_pool()
    cell = _cells(pool)[0]
    plan = {}
    for k in [int(x) for x in args.ks.split(",")]:
        members = order[:k]
        sub = {t: pool[t] for t in members}
        nrm = client_norms(sub, cell)
        spread = max(nrm.values()) / max(min(nrm.values()), 1e-12)
        tasks = [t for t in T.TASKS if T.TASKS[t][3] in members and T.TASKS[t][2].startswith("label_acc")]
        plan[k] = {"members": members, "tasks": tasks, "norm_spread_at_cell0": spread}
        print(f"k={k}: spread {spread:8.1f}x  members={members}")
        print(f"      tasks({len(tasks)}) = {','.join(tasks)}")
        for rule in args.rules.split(","):
            fn, kw = RULES[rule]
            n_r = k * next(iter(next(iter(sub.values())).values()))["A"].shape[0]
            if "rank" in kw and kw["rank"] > n_r:
                kw = {**kw, "rank": n_r}
                if "k_common" in kw or rule.startswith(("isocts", "adaptive")):
                    kw = {**kw, "k_common": max(1, n_r // 2)}
            t0 = time.time()
            merged = fn(sub, **kw)
            out_dir = os.path.join(OUT, f"{args.order}-k{k}-{rule}")
            desc = f"rule={rule} kwargs={kw} k={k} order={args.order} members={members} spread={spread:.1f}"
            if _unchanged(merged, out_dir):
                print(f"  {rule:16s} unchanged"); continue
            write_merged_adapter(merged, out_dir, merged_rank(merged), description=desc)
            print(f"  {rule:16s} rank={merged_rank(merged):3d} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(plan, open(os.path.join(OUT, "plan.json"), "w"), indent=1)
    print(f"\nplan -> {os.path.join(OUT, 'plan.json')}")

if __name__ == "__main__":
    main()
