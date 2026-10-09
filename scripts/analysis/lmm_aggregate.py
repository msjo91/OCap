import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

SEEN = ["ScienceQA", "ImageNet", "VQAv2", "REC", "OCRVQA", "VizWiz", "Flickr30k", "IconQA"]
UNSEEN = ["AOKVQA", "ImageNet-R", "Screen2words", "TabMWP"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=res("lmm_eval")); ap.add_argument("--models", default=None)
    args = ap.parse_args()
    rows = {}
    for f in sorted(glob.glob(f"{args.dir}/*__metrics.json")):
        m = os.path.basename(f)[:-14]
        if args.models and m not in args.models.split(","):
            continue
        rows[m] = {t: v for t, v in json.load(open(f)).items() if v.get("acc") is not None and v.get("n_skipped", 0) == 0}
    tasks = [t for t in SEEN + UNSEEN if any(t in v for v in rows.values())]
    common_seen = [t for t in SEEN if all(t in v for v in rows.values())]
    common_unseen = [t for t in UNSEEN if all(t in v for v in rows.values())]
    print(f"MM-MergeBench, n per task = {next(iter(rows.values()))[tasks[0]].get('n_requested')} (strict/norm for exact-match tasks); "
          f"means over the {len(common_seen)} seen + {len(common_unseen)} unseen tasks every model has")
    print(f"{'model':32s}" + "".join(f"{t[:10]:>12s}" for t in tasks) + f"{'seen':>7s}{'unseen':>7s}")
    for m, v in rows.items():
        cells = []
        for t in tasks:
            if t not in v:
                cells.append(f"{'-':>12s}"); continue
            a = v[t]["acc"] * 100; an = v[t].get("acc_norm")
            cells.append(f"{a:5.1f}/{an*100:5.1f} " if an is not None else f"{a:11.1f} ")
        s = np.mean([v[t]["acc"] for t in common_seen]) * 100 if common_seen else float("nan")
        u = np.mean([v[t]["acc"] for t in common_unseen]) * 100 if common_unseen else float("nan")
        sn = np.mean([v[t].get("acc_norm", v[t]["acc"]) for t in common_seen]) * 100 if common_seen else float("nan")
        un = np.mean([v[t].get("acc_norm", v[t]["acc"]) for t in common_unseen]) * 100 if common_unseen else float("nan")
        print(f"{m[:32]:32s}" + "".join(cells) + f"{s:7.1f}{u:7.1f}   norm {sn:5.1f}/{un:5.1f}")
    print("reference (RobustMerge Table 1, full sets, strict): zero-shot 43.4/25.2, TA 53.9/33.3, TIES 53.1/33.1, RobustMerge 57.3/38.0, individual 69.2")

if __name__ == "__main__":
    main()
