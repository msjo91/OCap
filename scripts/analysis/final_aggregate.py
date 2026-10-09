import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

C6 = ["fpb", "fiqa", "tfns", "nwgi", "headline", "ner"]
X6 = ["xbrl_finer", "xbrl_tags_extract", "xbrl_value_extract",
      "xbrl_formula_extract", "xbrl_formula_calc_extract", "formula"]
T12 = C6 + X6
HO = ["finqa", "convfinqa", "finexam10k"]

def load(d):
    out = {}
    for f in glob.glob(f"{d}/*__metrics.json"):
        out[os.path.basename(f).replace("__metrics.json", "")] = json.load(open(f))
    return out

def mean_se(v, tasks):
    got = [t for t in tasks if t in v and v[t].get("acc") is not None]
    if not got:
        return None, None, 0
    m = float(np.mean([v[t]["acc"] for t in got]))
    se = float(np.sqrt(sum(v[t]["acc"] * (1 - v[t]["acc"]) / v[t]["n"] for t in got if v[t].get("n"))) / len(got))
    return m, se, len(got)

def hdr(t):
    print(f"\n{'='*100}\n{t}\n{'='*100}")

def main_table():
    hdr("1. MAIN TABLE — public FinLoRA pool, n=2000 (retention 12 tasks | transfer 3 suites)")
    P, H = load(res("e1_pool2000")), load(res("e1_heldout2000"))
    E8 = load(res("e8_2000"))
    home = {}
    for m, v in E8.items():
        for t, r in v.items():
            if r.get("acc") is not None:
                home[t] = max(home.get(t, 0.0), r["acc"])
    rows = []
    for m, v in P.items():
        mm, se, n = mean_se(v, T12)
        if mm is None:
            continue
        c, _, _ = mean_se(v, C6); x, _, _ = mean_se(v, X6)
        hv = H.get(m, {})
        hm, hse, _ = mean_se(hv, HO)
        ret = np.mean([v[t]["acc"] / home[t] for t in T12 if t in v and home.get(t)]) if home else None
        ect = hv.get("ectsum", {}).get("rougeLsum")
        rows.append((m, mm, se, c, x, ret, hm, hse, ect, n))
    rows.sort(key=lambda r: -r[1])
    print(f"{'model':26s}{'pool':>8s}{'±SE':>7s}{'class6':>8s}{'xbrl6':>7s}{'reten':>7s}{'held':>8s}{'±SE':>7s}{'ECTSum':>8s}{'nT':>4s}")
    for m, mm, se, c, x, ret, hm, hse, ect, n in rows:
        print(f"{m.replace('merged_',''):26s}{mm:8.4f}{se:7.4f}{c:8.3f}{x:7.3f}"
              f"{(f'{ret:7.2f}' if ret else '      -')}"
              f"{(f'{hm:8.4f}' if hm else '       -')}{(f'{hse:7.4f}' if hse else '      -')}"
              f"{(f'{ect:8.3f}' if ect else '       -')}{n:4d}")
    if home:
        print(f"\noracle routing (per-task best client) over the 12 tasks: {np.mean([home[t] for t in T12 if t in home]):.4f} "
              f"({sum(t in home for t in T12)}/12 tasks measured at n=2000)")

def capability():
    hdr("2. CAPABILITY — lm-eval, common limit")
    rows = []
    for base in ("e6_2000", "e6_500", "e6"):
        for md in sorted(glob.glob(res(base, "*"))):
            if not os.path.isdir(md):
                continue
            agg = {"suite": base, "model": os.path.basename(md)}
            for f in glob.glob(f"{md}/**/results_*.json", recursive=True):
                d = json.load(open(f)); rs = d["results"]
                mm = [v["acc,none"] for k, v in rs.items() if k.startswith("mmlu_") and "acc,none" in v]
                if mm: agg["mmlu"] = float(np.mean(mm))
                for k, v in rs.items():
                    if k == "gsm8k":
                        for kk, vv in v.items():
                            if kk.startswith("exact_match,strict"): agg["strict"] = vv
                            if kk.startswith("exact_match,flex"): agg["flex"] = vv
                    if k == "hellaswag": agg["hs"] = v.get("acc_norm,none")
            if len(agg) > 2: rows.append(agg)
    rows.sort(key=lambda r: (r["suite"] != "e6_500", -(r.get("strict") or -1)))
    print(f"{'suite':9s}{'model':26s}{'MMLU':>8s}{'GSM8K-s':>9s}{'GSM8K-f':>9s}{'HellaSwag':>10s}")
    for r in rows:
        print(f"{r['suite']:9s}{r['model']:26s}"
              + "".join(f"{r.get(k):>9.3f}" if isinstance(r.get(k), float) else f"{'-':>9s}"
                        for k in ("mmlu", "strict", "flex", "hs")))

def sensitivity():
    hdr("3. SENSITIVITY OF THE NAMED RULE'S OWN CONSTANTS (n=500)")
    P, H = load(res("e1_pool500")), load(res("e1_heldout500"))
    fam = {"k_common": ["isocts_white_kc16", "isocts_white_kc24", "isocts_white", "isocts_white_kc40", "isocts_white_kc48"],
           "alpha": ["isocts_white_a05", "isocts_white_a075", "isocts_white", "isocts_white_a15", "isocts_white_a20"],
           "target": ["isocts_white", "isocts_white_mean"],
           "beta @ k_c=56": ["asym_shared_only", "asym_cts_b25_k56", "asym_cts_k56", "asym_cts_b75_k56", "asym_full"],
           "polar client step + operator": ["isocts_white", "knots_white", "ties_white", "tsvm_white", "dcmerge_white"]}
    for name, ms in fam.items():
        print(f"\n--- {name} ---")
        for m in ms:
            v = P.get("merged_" + m)
            if not v: print(f"   {m:24s} (pending)"); continue
            mm, se, n = mean_se(v, T12); hm, _, _ = mean_se(H.get("merged_" + m, {}), HO)
            print(f"   {m:24s} pool={mm:.4f} ±{se:.4f} ({n} tasks)  held={hm:.4f}" if hm else
                  f"   {m:24s} pool={mm:.4f} ±{se:.4f} ({n} tasks)")

def generality():
    hdr("4. GENERALITY — other pools, sizes and modalities")
    print("\n--- client-count sweep (n=200; each merge on its own clients' tasks) ---")
    N = load(res("nclients_pool"))
    got = {}
    for m, v in N.items():
        parts = m.replace("merged_", "").split("-")
        if len(parts) != 3: continue
        a, _, _ = mean_se(v, list(v))
        got[(parts[0], int(parts[1][1:]), parts[2])] = a
    rules = sorted({r for _, _, r in got})
    for order in ("light_first", "heavy_first"):
        ks = sorted({k for o, k, _ in got if o == order})
        if not ks: continue
        print(f"  {order}")
        for r in rules:
            cells = "".join(f"{got[(order,k,r)]:<9.3f}" if (order, k, r) in got else f"{'-':<9s}" for k in ks)
            if cells.strip("- "): print(f"    {r:22s}" + "".join(f"k={k:<7d}" for k in ks) if False else f"    {r:22s}{cells}")
        print(f"    {'(k =)':22s}" + "".join(f"{k:<9d}" for k in ks))
    print("\n--- rsLoRA sibling pool ---")
    RP, RH = load(res("rslora_pool")), load(res("rslora_heldout"))
    for m in sorted(RP):
        if not m.startswith("merged_"): continue
        mm, _, n = mean_se(RP[m], T12); hm, _, _ = mean_se(RH.get(m, {}), HO)
        print(f"   {m.replace('merged_',''):20s} pool={mm:.4f} ({n})  held={hm if hm else float('nan'):.4f}")

def diagnosis():
    hdr("5. DIAGNOSIS ACROSS POOLS — is A shared? is the pool dominated?")
    print(f"{'pool':36s}{'base':34s}{'n':>4s}{'spread':>9s}{'A overlap':>12s}{'B overlap':>12s}")
    for f, lab in [(res("d1", "d1_r8.json"), "FinLoRA r8 (independent A0)"),
                   (res("d1", "d1_rslora.json"), "FinLoRA rsLoRA"),
                   (res("d1", "d1_dora.json"), "FinLoRA DoRA"),
                   (res("d1", "predibase_mistral7b.json"), "Predibase LoRA Land"),
                   (res("d1", "mergemedbench.json"), "MergeMedBench")]:
        if not os.path.exists(f): print(f"{lab:36s}(pending)"); continue
        d = json.load(open(f)); s = d.get("summary", d)
        keys = [k for k in s if isinstance(s[k], dict) and "excess_A_sigma" in s[k]]
        aa = np.mean([s[k].get("sim_A", s[k].get("sim_A_mean")) for k in keys])
        bb = np.mean([s[k].get("sim_B", s[k].get("sim_B_mean")) for k in keys])
        sa = np.mean([s[k]["excess_A_sigma"] for k in keys]); sb = np.mean([s[k]["excess_B_sigma"] for k in keys])
        print(f"{lab:36s}{str(d.get('base',''))[:33]:34s}{d.get('n_clients',8):4d}"
              f"{(f'{d['spread']:8.1f}x' if d.get('spread') else '        -')}"
              f"{aa:7.3f}{sa:+6.0f}σ{bb:7.3f}{sb:+6.0f}σ")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--section", default="all")
    a = ap.parse_args()
    fns = {"main": main_table, "cap": capability, "sens": sensitivity, "gen": generality, "diag": diagnosis}
    for k, fn in fns.items():
        if a.section in ("all", k):
            try: fn()
            except Exception as e: print(f"\n[{k} failed: {type(e).__name__}: {e}]")
