import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import tasks as T

LAB = [t for t, (f, _, m, o) in T.TASKS.items() if m.startswith("label_acc")]
_LABELS = {}

def rows(d, model, t):
    f = f"{d}/{model}__{t}.jsonl"
    if not os.path.exists(f):
        return None
    if t not in _LABELS:
        _LABELS[t] = T.load_task(t, n=None)[1]
    r = [json.loads(l) for l in open(f)]; o = [x["output"] for x in r]; g = [x["target"] for x in r]
    if T.TASKS[t][2] == "label_acc_batched":
        o, g = T.split_batched(o, g)
    return np.array([int(T.score_task(t, [a], [b], _LABELS[t])["acc"]) for a, b in zip(o, g)])

def task_balanced(diffs, n_boot=2000, seed=0):
    
    rng = np.random.default_rng(seed)
    ts = list(diffs)
    point = float(np.mean([diffs[t].mean() for t in ts]))
    bs = []
    for _ in range(n_boot):
        bs.append(np.mean([diffs[t][rng.integers(0, len(diffs[t]), len(diffs[t]))].mean() for t in ts]))
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def compare(d, a, b, tasks=None, d_b=None):
    diffs = {}
    for t in tasks or LAB:
        ra, rb = rows(d, a, t), rows(d_b or d, b, t)
        if ra is not None and rb is not None and len(ra) == len(rb):
            diffs[t] = ra - rb
    return task_balanced(diffs), sorted(diffs)

if __name__ == "__main__":
    d, a, b = sys.argv[1:4]
    tasks = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    (m, lo, hi), ts = compare(d, a, b, tasks)
    print(f"{a} - {b}: task-balanced {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(ts)} tasks")
