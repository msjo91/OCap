import argparse, glob, json, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import RESULT_DIRS, res, res_rel

ENTRYWISE = ("ties", "dare", "trim", "elect", "knots")
SKIP_DIRS = {"launch", "merges", "merges_loo", "merges_rslora", "merges_nclients", "n5",
             "n5_merges", "lmm_merges", "d1", "e6", "e6_500", "e6_2000", "e6_500_rslora"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dirs", default=None)
    a = ap.parse_args()
    names = a.dirs.split(",") if a.dirs else [
        n for n in sorted(RESULT_DIRS) if n not in SKIP_DIRS and os.path.isdir(res(n))
        and glob.glob(res(n, "*__metrics.json"))]
    issues = defaultdict(list)

    for name in names:
        known = name in RESULT_DIRS
        path, d = (res(name), res_rel(name)) if known else (name, name)
        models = {}
        for f in glob.glob(f"{path}/*__metrics.json"):
            m = os.path.basename(f).replace("__metrics.json", "")
            try: models[m] = json.load(open(f))
            except Exception as e: issues["2 FAILED (unreadable metrics)"].append(f"{f}: {e}")
        if not models: continue
        counts = defaultdict(int)
        for v in models.values():
            for t, r in v.items():
                if isinstance(r, dict) and r.get("acc") is not None: counts[t] += 1
        quorum = max(counts.values()) if counts else 0
        core = {t for t, c in counts.items() if c >= max(2, 0.6 * quorum)}
        for m, v in models.items():
            have = {t for t, r in v.items() if isinstance(r, dict) and r.get("acc") is not None}
            miss = core - have
            by_design = any(k in name for k in ("loo_pool", "nclients_pool", "e8", "rho_pool", "n5_"))
            if miss and len(have) > 0 and not by_design:
                issues["1 MISSING/PARTIAL"].append(f"{d}/{m}: missing {len(miss)}/{len(core)} -> {sorted(miss)[:5]}")

        if "loo_pool" in name:
            groups = defaultdict(dict)
            for m, v in models.items():
                mm = re.match(r"^(.+)-no_(.+)$", m)
                if mm:
                    groups[mm.group(2)][mm.group(1)] = {
                        t for t, r in v.items() if isinstance(r, dict) and r.get("acc") is not None}
            for client, per_rule in groups.items():
                if len(per_rule) < 2:
                    continue
                sizes = defaultdict(int)
                for ts in per_rule.values():
                    sizes[frozenset(ts)] += 1
                ref = max(sizes, key=lambda k: (sizes[k], len(k)))
                for rule, ts in sorted(per_rule.items()):
                    if ts != ref:
                        issues["1 MISSING/PARTIAL"].append(
                            f"{d}/{rule}-no_{client}: {len(ts)}/{len(ref)} tasks of that client "
                            f"-> missing {sorted(ref - ts)[:5]}")
        for t in core:
            ns = defaultdict(list); gc = defaultdict(list)
            for m, v in models.items():
                r = v.get(t)
                if isinstance(r, dict) and r.get("acc") is not None:
                    ns[str(r.get("n"))].append(m); gc[str(r.get("gen_cfg"))].append(m)
            if len(ns) > 1:
                issues["3 UNFAIR-N"].append(f"{d}:{t} -> " + "; ".join(f"n={k}:{len(v)} models" for k, v in ns.items()))
            if len(gc) > 1:
                issues["4 UNFAIR-PROMPT"].append(f"{d}:{t} -> " + "; ".join(f"{str(k)[:34]}:{len(v)}" for k, v in gc.items()))

    for lg in glob.glob(res("logs", "*.log")):
        try: txt = open(lg, errors="ignore").read()
        except Exception: continue
        for pat in ("Traceback (most recent call last)", "CUDA out of memory", "CUDA error"):
            if pat in txt:
                issues["2 FAILED (log)"].append(f"{res_rel('logs', os.path.basename(lg))}: {pat} x{txt.count(pat)}"); break
        if re.search(r"--model merged:\$|--model merged:\s", txt):
            issues["2 FAILED (unexpanded variable in job)"].append(res_rel("logs", os.path.basename(lg)))

    for info in sorted(f for n in RESULT_DIRS if n.startswith("merges") for f in glob.glob(res(n, "*", "MERGE_INFO.txt"))):
        txt = open(info).read(); name = os.path.basename(os.path.dirname(info))
        if any(k in name for k in ENTRYWISE): continue
        n = re.search(r"\bn=(\d+)", txt); k = re.search(r"\bk=(\d+)", txt)
        rk = re.search(r"'rank':\s*(\d+)", txt)
        nn = int(k.group(1)) if k else (int(n.group(1)) if n else None)
        if not (nn and rk): continue
        r = 16 if "lmm" in info else 8
        if int(rk.group(1)) > nn * r:
            issues["5 MISSPEC (rank > n*r)"].append(f"{os.path.dirname(info)}: rank={rk.group(1)} n*r={nn*r}")

    tot = 0
    for k in sorted(issues):
        print(f"\n### {k}: {len(issues[k])}")
        for x in issues[k][:10]: print("   ", x)
        if len(issues[k]) > 10: print(f"    ... and {len(issues[k])-10} more")
        tot += len(issues[k])
    print(f"\n{'NO ISSUES FOUND' if tot == 0 else f'TOTAL ISSUES: {tot}'}")

if __name__ == "__main__":
    main()
