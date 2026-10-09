import json
import os
import random
import re

from .paths import ROOT

HELDOUT_DATA = os.path.join(ROOT, "data", "heldout")

TASKS = {
    "finqa":      {"max_new_tokens": 32,  "metric": "numeric", "max_length": 4096, "batch_size": 4},
    "convfinqa":  {"max_new_tokens": 32,  "metric": "numeric", "max_length": 4096, "batch_size": 4},
    "finexam10k": {"max_new_tokens": 8,   "metric": "letter",  "max_length": 2048, "batch_size": 8},
    "ectsum":     {"max_new_tokens": 160, "metric": "rouge",   "max_length": 16384, "batch_size": 1},
}

def load_task(name, n=None, seed=42, clean=False):
    
    fn = os.path.join(HELDOUT_DATA, name, "test.jsonl")
    rows = [json.loads(l) for l in open(fn)]
    if clean:
        from .tasks import _norm_ctx, _train_contexts
        tr = _train_contexts(); seen = set(); keep = []
        for r in rows:
            k = _norm_ctx(r["context"])
            if k in tr or k in seen:
                continue
            seen.add(k); keep.append(r)
        rows = keep
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    for r in rows:
        ctx = r["context"].rstrip()
        if not ctx.endswith(":"):
            ctx += "\nAnswer:"
        r["context"] = ctx
    return rows

_NUM_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")

def parse_number(text):
    
    m = _NUM_RE.search(str(text))
    if not m:
        return None
    s = m.group(0).replace("$", "").replace(",", "")
    neg = s.startswith("(") and s.rstrip("%").endswith(")")
    pct = s.endswith("%")
    s = s.strip("()%").rstrip(")")
    try:
        v = float(s)
    except ValueError:
        return None
    if neg:
        v = -v
    if pct:
        v = v / 100.0
    return v

def numeric_match(output, target, rel_tol=0.01):
    
    t = str(target).strip().lower()
    o = str(output).strip().lower()
    if t in ("yes", "no"):
        m = re.search(r"\b(yes|no)\b", o)
        return bool(m and m.group(1) == t)
    tv, ov = parse_number(t), parse_number(o)
    if tv is None or ov is None:
        return False
    for cand in (ov, ov / 100.0, ov * 100.0):
        denom = max(abs(tv), 1e-8)
        if abs(cand - tv) / denom <= rel_tol:
            return True
    return False

def letter_match(output, target):
    m = re.search(r"\b([A-D])\b", str(output).upper())
    return bool(m and m.group(1) == str(target).strip().upper())

def score(name, outputs, targets):
    metric = TASKS[name]["metric"]
    if metric == "numeric":
        hits = [numeric_match(o, t) for o, t in zip(outputs, targets)]
    elif metric == "letter":
        hits = [letter_match(o, t) for o, t in zip(outputs, targets)]
    else:
        return {"acc": None, "n": len(outputs), "note": "score offline (rouge)"}
    return {"acc": sum(hits) / len(hits) if hits else 0.0, "n": len(hits)}
