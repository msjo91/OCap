import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src.pool import load_pool, write_merged_adapter
from src.merge import merge_isocts, merge_concat, whiten_pool, norm_match, merged_rank
from scripts.merge.make_merges import _unchanged

LIGHT = ["financebench", "formula", "xbrl_extract", "xbrl_term", "finer", "ner", "headline", "sentiment"]
OUT = res("merges_nclients")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ks", default="2,3,4,6,8")
    args = ap.parse_args()
    full = load_pool(pool="r8")
    heavy = LIGHT[::-1]
    for k in [int(x) for x in args.ks.split(",")]:
        sub = {t: full[t] for t in heavy[:k]}
        n_r = k * 8
        w = whiten_pool(sub, target="median")
        merged = merge_isocts(w, rank=n_r, k_common=n_r // 2)
        ref = merge_concat(w)
        merged = norm_match(merged, ref)
        out_dir = os.path.join(OUT, f"heavy_first-k{k}-isocts_white_nm")
        desc = (f"rule=isocts_white_nm kwargs={{'rank': {n_r}, 'k_common': {n_r//2}, 'ref': 'concat_white'}} "
                f"k={k} order=heavy_first members={heavy[:k]}")
        if _unchanged(merged, out_dir):
            print(f"k={k}: unchanged"); continue
        write_merged_adapter(merged, out_dir, merged_rank(merged), description=desc)
        cell = (16, "q_proj")
        print(f"k={k}: rank={merged_rank(merged)} "
              f"||dW||={np.linalg.norm(merged[cell]['B'] @ merged[cell]['A']):.3f} -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
