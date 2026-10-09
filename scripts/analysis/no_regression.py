import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from scripts.analysis import task_bootstrap as TB
from scripts.analysis import heldout_bootstrap as HB

POOL, HELD, CAP = res("e1_pool2000"), res("e1_heldout2000"), res("e6_500")
SUITES = ["finqa", "convfinqa", "finexam10k"]
N_BOOT = 2000

def interval(point, boots):
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def transfer(a, b="base"):
    rng = np.random.default_rng(0)
    per = {}
    for t in SUITES:
        ha, hb = HB.hits(HELD, a, t), HB.hits(HELD, b, t)
        if not ha or not hb:
            return None
        shared = sorted(set(ha) & set(hb))
        per[t] = np.array([ha[i] - hb[i] for i in shared])
    point = np.mean([v.mean() for v in per.values()])
    boots = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in per.values()]) for _ in range(N_BOOT)]
    return interval(point, boots)

def gsm8k_items(stem):
    fs = sorted(glob.glob(os.path.join(CAP, stem, "**", "samples_gsm8k_*.jsonl"), recursive=True))
    if not fs:
        return None
    out = {"strict-match": {}, "flexible-extract": {}}
    for line in open(fs[-1]):
        r = json.loads(line)
        if r.get("filter") in out:
            out[r["filter"]][r["doc_id"]] = float(r["exact_match"])
    return out

def gsm8k(stem, base="base"):
    a, b = gsm8k_items(stem), gsm8k_items(base)
    if a is None or b is None:
        return None
    res = {}
    for f in ("strict-match", "flexible-extract"):
        ids = sorted(set(a[f]) & set(b[f]))
        d = np.array([a[f][i] - b[f][i] for i in ids])
        rng = np.random.default_rng(0)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
        res[f] = interval(d.mean(), boots)
    return res

def z(iv):
    p, lo, hi = iv
    return p / ((hi - lo) / 3.92) if hi > lo else float("inf") * np.sign(p)

def rules_with_all_axes():
    have = []
    for f in sorted(glob.glob(os.path.join(POOL, "merged_*__metrics.json"))):
        m = os.path.basename(f)[:-len("__metrics.json")]
        stem = m[len("merged_"):]
        if not os.path.exists(os.path.join(HELD, f"{m}__metrics.json")):
            continue
        if not os.path.exists(os.path.join(CAP, stem, "done")):
            continue
        pool = json.load(open(f)); pool = pool.get("tasks", pool)
        held = json.load(open(os.path.join(HELD, f"{m}__metrics.json"))); held = held.get("tasks", held)
        if sum(isinstance(v, dict) for v in pool.values()) < 12 or not all(t in held for t in SUITES):
            continue
        have.append(m)
    return have

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=res("no_regression.json"))
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    cache = json.load(open(args.json)) if os.path.exists(args.json) else {}
    for m in rules_with_all_axes():
        if m in cache:
            continue
        stem = m[len("merged_"):]
        (rp, rlo, rhi), _ = TB.compare(POOL, m, "base")
        tr = transfer(m)
        g = gsm8k(stem)
        if tr is None or g is None:
            print(f"skip {m}: incomplete", file=sys.stderr)
            continue
        cache[m] = {"retention": [rp, rlo, rhi], "transfer": list(tr),
                    "gsm8k_strict": list(g["strict-match"]), "gsm8k_flexible": list(g["flexible-extract"])}
        json.dump(cache, open(args.json, "w"), indent=1)
        print(f"{m}: done", file=sys.stderr)
    rows = []
    for m, v in cache.items():
        zs = {k: z(v[k]) for k in ("retention", "transfer", "gsm8k_strict", "gsm8k_flexible")}
        rows.append((min(zs.values()), m, v, zs))
    rows.sort(key=lambda r: -r[0])
    if args.latex:
        emit_latex(rows)
        return
    for wz, m, v, zs in rows:
        cells = " ".join(f"{k}={v[k][0]:+.3f}[{v[k][1]:+.3f},{v[k][2]:+.3f}]" for k in v)
        print(f"{m:32s} worst_z={wz:+.2f}  {cells}")

NAME = {
    "adaptive_mad": r"PolCap$+$Iso-CTS",
    "adaptive_tinf": r"PolCap$+$Iso-CTS, $\tau{=}\infty$ (cap removed)",
    "dcmerge_g50": r"DC-Merge, $\gamma{=}.5$",
    "isocts_white": r"PolEq$+$Iso-CTS",
    "isoc_white": r"PolEq$+$Iso-C",
    "asym_cts": r"PolEq$+$asym-CTS, $k_c{=}32$",
    "asym_cts_k16": r"PolEq$+$asym-CTS, $k_c{=}16$",
    "asym_cts_k48": r"PolEq$+$asym-CTS, $k_c{=}48$",
    "asym_cts_k56": r"PolEq$+$asym-CTS, $k_c{=}56$",
    "isocts_neq": r"Iso-CTS, full equalization",
    "isocts_neq_nm": r"Iso-CTS, full equalization, TA norm",
    "isocts_g50": r"Iso-CTS, $\gamma{=}.5$",
    "isocts_g60": r"Iso-CTS, $\gamma{=}.6$",
    "isocts_g55": r"Iso-CTS, $\gamma{=}.55$",
    "isocts_cap25": r"Iso-CTS, share cap $\frac14$",
    "concat": "task arithmetic",
    "adaptive_mad_dcmerge": r"PolCap$+$DC-Merge",
    "adaptive_mad_concat": r"PolCap$+$task arithmetic",
    "adaptive_mad_k2": r"PolCap$+$Iso-CTS, $2$ MADs",
    "capmad_isocts": r"\ours{}$+$Iso-CTS",
    "capmad_concat": r"\textbf{\ours{}$+$task arithmetic (ours)}",
    "capmad_dcmerge": r"\ours{}$+$DC-Merge",
    "capmed_concat": r"clip at median$+$task arithmetic",
    "concat_nm_capmad": r"task arithmetic, shrunk to \ours{}'s norm",
    "domerging": r"DO-Merging, $\lambda{=}1/n^2$",
    "normeq_med": r"task arithmetic, full equalization",
    "ties64_g50": r"TIES, $\gamma{=}.5$",
    "trim_g50": r"TIES-trim, $\gamma{=}.5$",
    "knots_ties_g50": r"KnOTS-TIES, $\gamma{=}.5$",
    "tsvm_g50": r"TSV-M, $\gamma{=}.5$",
}
KEYS = ("retention", "transfer", "gsm8k_strict", "gsm8k_flexible")

def cellx(iv):
    p, lo, hi = iv
    body = f"{p:+.3f}".replace("0.", ".")
    ci = f"[{lo:+.3f},{hi:+.3f}]".replace("0.", ".")
    if hi < 0:
        body = r"\mathbf{" + body + "}"
    return f"${body}$\\,{{\\tiny${ci}$}}"

def emit_latex(rows):
    
    clean = [r for r in rows if all(r[2][k][2] >= 0 for k in KEYS)]
    for i, (wz, m, v, zs) in enumerate(rows):
        stem = m[len("merged_"):]
        name = NAME.get(stem, r"\texttt{" + stem.replace("_", r"\_") + "}")
        mark = r"\;$\checkmark$" if all(v[k][2] >= 0 for k in KEYS) else ""
        print(f"{name}{mark} & " + " & ".join(cellx(v[k]) for k in KEYS) + f" & ${wz:+.2f}$ \\\\")
        if i + 1 == len(clean):
            print(r"\midrule")

if __name__ == "__main__":
    main()
