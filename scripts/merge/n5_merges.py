import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src.pool import load_local_adapter_factors, load_adapter_factors, write_merged_adapter
from src.merge import RULES, regime_info, merged_rank, merge_regime_switch, _delta, _cells
from scripts.merge.make_merges import _unchanged

N5_DIR = res("n5")
OUT_DIR = res("n5_merges")
N5_TASKS = ["sentiment", "headline", "ner"]
LIGHT = ["finer", "xbrl_extract", "xbrl_term", "formula", "financebench"]
DEFAULT_RULES = ["concat", "trim_g50", "isocts_g50", "isocts_neq", "isocts_white_s3", "isoc_white_s3", "knots_ties_g50", "avg_factors", "avg_factors_nm", "regime_switch", "regime_switch_t98"]
REGIMES = {"S": "shared A_0, full recipe", "I": "independent A_0, full recipe",
           "S_fz": "shared A_0, A frozen (zero drift)", "S_lr01": "shared A_0, A lr x0.1 (small drift)",
           "I2": "independent A_0, full recipe, SECOND draw (replication of the I regime, 2026-09-18)"}

def is_self_trained(member):
    return any(member.endswith("_" + r) for r in REGIMES)

def member_task(member):
    
    for r in sorted(REGIMES, key=len, reverse=True):
        if member.endswith("_" + r):
            return member[: -len(r) - 1]
    return member

def pool_members(name):
    if name in REGIMES:
        return [f"{t}_{name}" for t in N5_TASKS]
    if name.endswith("_light") and name[:-6] in REGIMES:
        return pool_members(name[:-6]) + LIGHT
    if name == "mixed":
        return pool_members("S") + pool_members("I") + LIGHT
    if name == "S_light":
        return pool_members("S") + LIGHT
    if name == "I_light":
        return pool_members("I") + LIGHT
    raise ValueError(name)

def load_member(name, cache):
    if name not in cache:
        if is_self_trained(name):
            cache[name] = load_local_adapter_factors(os.path.join(N5_DIR, name), label=name)[0]
        else:
            cache[name] = load_adapter_factors(name)[0]
    return cache[name]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="S,I,mixed,S_light,I_light")
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES))
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    cache, detector = {}, {}
    for pname in args.pools.split(","):
        members = pool_members(pname)
        pool = {m: load_member(m, cache) for m in members}
        cells = _cells(pool)
        norms = {m: float(np.median([np.linalg.norm(_delta(pool[m][c])) for c in cells])) for m in members}
        info = regime_info(pool, tau=args.tau)
        detector[pname] = {"members": members, "median_dW_norm": norms, **info}
        print(f"\n=== pool {pname}: {len(members)} clients; median ||dW|| per client: "
              + ", ".join(f"{m} {v:.3f}" for m, v in norms.items()))
        print(f"A-frame similarity (chance {info['chance_mean']:.3f} ± {info['chance_sd']:.3f}; tau={args.tau}):")
        names, S = info["names"], np.array(info["sim"])
        w = max(len(n) for n in names)
        print(" " * (w + 1) + " ".join(f"{n[:7]:>7s}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:>{w}s} " + " ".join(f"{S[i, j]:7.3f}" for j in range(len(names))))
        print(f"clusters: {info['clusters']}  (max below tau {info['max_below']}, min above tau {info['min_above']})")
        for rule in args.rules.split(","):
            fn, kw = RULES[rule]
            if rule == "regime_switch":
                kw = {**kw, "tau": args.tau}
            rule_info = regime_info(pool, tau=kw["tau"]) if fn is merge_regime_switch else None
            t0 = time.time()
            merged = fn(pool, **kw)
            for cell, fs in merged.items():
                assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), f"{pname}/{rule}: non-finite at {cell}"
                assert np.abs(fs["B"]).max() > 0, f"{pname}/{rule}: all-zero B at {cell}"
            rank = merged_rank(merged)
            mnorm = float(np.median([np.linalg.norm(_delta(merged[c])) for c in cells]))
            out_dir = os.path.join(OUT_DIR, f"{pname}-{rule}")
            desc = (f"rule={rule} kwargs={kw} pool={pname} n={len(members)} members={members} "
                    f"clusters={rule_info['clusters'] if rule_info else 'n/a'} rank={rank}")
            if _unchanged(merged, out_dir):
                with open(os.path.join(out_dir, "MERGE_INFO.txt"), "w") as fh:
                    fh.write(desc + "\n")
                print(f"  {rule:14s} rank={rank:3d} median||dW||={mnorm:.3f} unchanged (identical factors on disk)", flush=True)
                continue
            write_merged_adapter(merged, out_dir, rank, description=desc)
            print(f"  {rule:14s} rank={rank:3d} median||dW||={mnorm:.3f} {time.time()-t0:.1f}s -> {out_dir}"
                  + (f"  clusters(tau={kw['tau']})={rule_info['clusters']}" if rule_info else ""), flush=True)
        with open(os.path.join(OUT_DIR, "detector.json"), "w") as f:
            json.dump(detector, f, indent=1)
    print(f"\ndetector output -> {os.path.join(OUT_DIR, 'detector.json')}")

if __name__ == "__main__":
    main()
