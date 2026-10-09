import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import heldout as H

RNG = np.random.default_rng(0)
DEFAULT = ["finqa", "convfinqa", "finexam10k"]

def hits(d, model, t):
    f = f"{d}/{model}__{t}.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    metric = H.TASKS[t]["metric"]
    fn = H.numeric_match if metric == "numeric" else H.letter_match
    key = "idx" if "idx" in rows[0] else "id"
    return {r[key]: int(fn(r["output"], r["target"])) for r in rows}

def main():
    d, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    suites = sys.argv[4].split(",") if len(sys.argv) > 4 else DEFAULT
    per = {}
    for t in suites:
        ha, hb = hits(d, a, t), hits(d, b, t)
        if not ha or not hb:
            continue
        shared = sorted(set(ha) & set(hb))
        if shared:
            per[t] = np.array([ha[i] - hb[i] for i in shared])
    if not per:
        print(f"{a} - {b}: no shared suites"); return
    point = np.mean([v.mean() for v in per.values()])
    bs = [np.mean([v[RNG.integers(0, len(v), len(v))].mean() for v in per.values()]) for _ in range(2000)]
    print(f"{a} - {b}: suite-balanced {point:+.3f} "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] over {len(per)} suites")

if __name__ == "__main__":
    main()
