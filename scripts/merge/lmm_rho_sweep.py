import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import ROOT, res
from src import lmm_pool as LP
from src.merge import RULES, normalize_pool, merged_rank, client_norms, _cells
from src import asym
from scripts.merge.make_merges import _unchanged

OUT = res("lmm_merges_rho")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", default="1,10,100")
    ap.add_argument("--client", default="OCRVQA")
    ap.add_argument("--rules", default="isocts,concat")
    ap.add_argument("--root", default=os.path.join(ROOT, "external", "mm_mergebench"))
    args = ap.parse_args()

    factors, projs = LP.load_llava_pool(args.root)
    projector = LP.mean_projector(projs)
    base_pool = normalize_pool(factors, "median", gamma=1.0, max_up=None)
    cell = _cells(base_pool)[0]
    print(f"pool: {len(base_pool)} clients, {len(_cells(base_pool))} cells; scaling '{args.client}'")

    for rho in [float(x) for x in args.rhos.split(",")]:
        pool = {t: {c: {"A": fs["A"], "B": fs["B"] * (rho if t == args.client else 1.0)}
                    for c, fs in base_pool[t].items()} for t in base_pool}
        nrm = client_norms(pool, cell)
        print(f"\nrho={rho:g}: {args.client} {nrm[args.client]:.3f} vs median others "
              f"{np.median([v for t, v in nrm.items() if t != args.client]):.3f}")
        for rule in args.rules.split(","):
            fn, kw = RULES[rule]
            n_r = len(pool) * next(iter(next(iter(pool.values())).values()))["A"].shape[0]
            if "rank" in kw and kw["rank"] < n_r:
                kw = {**kw, "rank": n_r}
            t0 = time.time()
            merged = fn(pool, **kw)
            rank = merged_rank(merged)
            out_dir = os.path.join(OUT, f"{rule}-rho{rho:g}")
            desc = (f"rule={rule} kwargs={kw} rho={rho:g} scaled_client={args.client} "
                    f"base=equalized(median) pool=mm_mergebench rank={rank}")
            if _unchanged(merged, out_dir):
                with open(os.path.join(out_dir, "MERGE_INFO.txt"), "w") as fh:
                    fh.write(desc + "\n")
                print(f"  {rule:10s} unchanged"); continue
            LP.write_llava_merged(merged, out_dir, rank, projector=projector, description=desc)
            print(f"  {rule:10s} rank={rank} {time.time()-t0:.0f}s -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
