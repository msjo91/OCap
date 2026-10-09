import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src.pool import load_pool, write_merged_adapter
from src.merge import RULES, normalize_pool, merged_rank, client_norms
from scripts.merge.make_merges import _unchanged

OUT = res("merges_rho")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", default="1,3,10,30,100,300,700"); ap.add_argument("--client", default="sentiment")
    ap.add_argument("--rules", default="isocts,concat"); args = ap.parse_args()
    base_pool = normalize_pool(load_pool(), "median", gamma=1.0, max_up=None)
    cell = (16, "q_proj")
    for rho in [float(x) for x in args.rhos.split(",")]:
        pool = {t: {c: {"A": fs["A"], "B": fs["B"] * (rho if t == args.client else 1.0)} for c, fs in base_pool[t].items()}
                for t in base_pool}
        n = client_norms(pool, cell); print(f"rho={rho:g}: {args.client} {n[args.client]:.3f} vs median others {np.median([v for t, v in n.items() if t != args.client]):.3f}")
        for rule in args.rules.split(","):
            fn, kw = RULES[rule]; t0 = time.time(); merged = fn(pool, **kw)
            out_dir = os.path.join(OUT, f"{rule}-rho{rho:g}")
            rank = merged_rank(merged)
            desc = f"rule={rule} kwargs={kw} rho={rho:g} scaled_client={args.client} base=equalized(median) pool=r8 rank={rank}"
            if _unchanged(merged, out_dir):
                with open(os.path.join(out_dir, "MERGE_INFO.txt"), "w") as fh:
                    fh.write(desc + "\n")
                print(f"  {rule:8s} unchanged"); continue
            write_merged_adapter(merged, out_dir, rank, description=desc)
            print(f"  {rule:8s} rank={rank} {time.time()-t0:.0f}s -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
