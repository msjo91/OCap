import argparse, glob, json, os, sys
import numpy as np
from rouge_score import rouge_scorer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=res("e1_heldout2000"))
    args = ap.parse_args()
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    for f in sorted(glob.glob(f"{args.dir}/*__ectsum.jsonl")):
        model = os.path.basename(f)[: -len("__ectsum.jsonl")]
        rows = [json.loads(l) for l in open(f)]
        r1, r2, rl = [], [], []
        for r in rows:
            s = sc.score(r["target"].strip(), r["output"].strip())
            r1.append(s["rouge1"].fmeasure); r2.append(s["rouge2"].fmeasure); rl.append(s["rougeLsum"].fmeasure)
        mf = os.path.join(args.dir, f"{model}__metrics.json")
        m = json.load(open(mf)) if os.path.exists(mf) else {}
        rec = m.get("ectsum", {})
        rec.update({"model": model, "task": "ectsum", "metric": "rouge", "n": len(rows),
                    "rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)),
                    "rougeLsum": float(np.mean(rl)),
                    "acc": float(np.mean(rl)),
                    "scorer": "rouge_score/use_stemmer, F-measure vs the single reference"})
        m["ectsum"] = rec
        tmp = mf + ".tmp"
        json.dump(m, open(tmp, "w"), indent=1); os.replace(tmp, mf)
        print(f"{model:34s} n={len(rows):4d}  R1={np.mean(r1):.4f}  R2={np.mean(r2):.4f}  RLsum={np.mean(rl):.4f}")

if __name__ == "__main__":
    main()
