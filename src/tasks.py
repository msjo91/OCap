import json
import re
import os
import random

from .paths import ROOT

FINLORA_DATA = os.path.join(ROOT, "external", "FinLoRA", "data", "test")

TASKS = {
    "fpb":               ("fpb_test.jsonl",                       10, "label_acc", "sentiment"),
    "fiqa":              ("fiqa_test.jsonl",                      10, "label_acc", "sentiment"),
    "tfns":              ("tfns_test.jsonl",                      10, "label_acc", "sentiment"),
    "nwgi":              ("nwgi_test.jsonl",                      10, "label_acc", "sentiment"),
    "headline":          ("headline_test.jsonl",                  10, "label_acc", "headline"),
    "ner":               ("ner_test.jsonl",                       10, "label_acc", "ner"),
    "xbrl_finer":        ("finer_test_batched.jsonl",            100, "label_acc_batched", "finer"),
    "xbrl_tags_extract": ("xbrl_extract_tags_test.jsonl",         20, "label_acc", "xbrl_extract"),
    "xbrl_value_extract": ("xbrl_extract_value_test.jsonl",       20, "label_acc", "xbrl_extract"),
    "xbrl_formula_extract": ("xbrl_extract_formula_test.jsonl",   30, "label_acc", "xbrl_extract"),
    "xbrl_formula_calc_extract":
        ("xbrl_extract_formula_calculations_test.jsonl",          30, "label_acc", "xbrl_extract"),
    "xbrl_term":         ("xbrl_term_test.jsonl",                 50, "bertscore", "xbrl_term"),
    "formula":           ("formula_test.jsonl",                   50, "label_acc", "formula"),
    "financebench":      ("financebench_test.jsonl",              50, "bertscore", "financebench"),
}

DELIMITER = "Answer:"

def client_tasks(client):
    return [name for name, (_, _, _, owner) in TASKS.items() if owner == client]

_CLEAN_CACHE = {}
_TRAIN_DIR = os.path.join(ROOT, "external", "FinLoRA", "data", "train")
_TRAIN_FILES = ["finlora_sentiment_train.jsonl", "headline_train.jsonl", "ner_train.jsonl",
                "finer_train_batched.jsonl", "xbrl_term_train.jsonl", "formula_train.jsonl",
                "financebench_train.jsonl"]

def _norm_ctx(s):
    import hashlib
    return hashlib.md5(re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s).lower()).strip()).encode()).hexdigest()

def _train_contexts():
    if "ctx" not in _CLEAN_CACHE:
        S = set()
        for f in _TRAIN_FILES:
            p = os.path.join(_TRAIN_DIR, f)
            if not os.path.exists(p):
                continue
            for line in open(p):
                try:
                    S.add(_norm_ctx(json.loads(line).get("context", "")))
                except Exception:
                    pass
        csv_dir = os.path.join(_TRAIN_DIR, "xbrl_csv")
        if os.path.isdir(csv_dir):
            import csv as _csv
            _csv.field_size_limit(10 ** 9)
            for c in sorted(os.listdir(csv_dir)):
                if c.endswith(".csv"):
                    with open(os.path.join(csv_dir, c), newline="") as fh:
                        for r in _csv.DictReader(fh):
                            S.add(_norm_ctx(r.get("input") or ""))
        _CLEAN_CACHE["ctx"] = S
    return _CLEAN_CACHE["ctx"]

def clean_filter(rows):
    
    train = _train_contexts()
    out, seen = [], set()
    for r in rows:
        k = _norm_ctx(r["context"])
        if k in train or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def load_task(name, n=None, seed=42, clean=False):
    
    fname, _, metric, _ = TASKS[name]
    rows = [json.loads(l) for l in open(os.path.join(FINLORA_DATA, fname))]
    for i, r in enumerate(rows):
        r["_idx"] = i
        r["target"] = str(r["target"])
    labels = None
    if metric == "label_acc":
        labels = sorted({r["target"] for r in rows})
    elif metric == "label_acc_batched":
        labels = sorted({p.strip().replace("\n", "")
                         for r in rows for p in r["target"].split(",")})
    if clean:
        rows = clean_filter(rows)
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows, labels

_LABEL_RE_CACHE = {}

def _label_regexes(labels):
    
    key = tuple(labels)
    if key not in _LABEL_RE_CACHE:
        alt = "|".join(re.escape(l.lower()) for l in sorted(labels, key=len, reverse=True))
        _LABEL_RE_CACHE[key] = {
            l: re.compile(r"(?<!\w)" + re.escape(l.lower()) + r"(?=$|[^a-z]|" + alt + r")")
            for l in labels}
    return _LABEL_RE_CACHE[key]

def first_occurrence_pred(output, labels):
    
    o = str(output).lower()
    hits = []
    for l, rx in _label_regexes(labels).items():
        m = rx.search(o)
        if m:
            hits.append((m.start(), -len(l), l))
    return min(hits)[2] if hits else None

def split_batched(outputs, targets):
    
    outs, tgts = [], []
    for o, t in zip(outputs, targets):
        so = [x.strip().replace("\n", "") for x in str(o).split(",")]
        st = [x.strip().replace("\n", "") for x in str(t).split(",")]
        tgts += st
        outs += so[:len(st)] + [""] * max(0, len(st) - len(so))
    return outs, tgts

NUMERIC_TASKS = {"xbrl_value_extract": {}, "formula": {"strip_percent": True}}

def score_numeric(outputs, targets, rel_tol=1e-6, strip_percent=False):
    
    from .heldout import parse_number
    hits = []
    for o, t in zip(outputs, targets):
        if strip_percent:
            o, t = str(o).replace("%", ""), str(t).replace("%", "")
        tv, ov = parse_number(t), parse_number(o)
        hits.append(int(tv is not None and ov is not None and abs(ov - tv) <= rel_tol * max(1.0, abs(tv))))
    acc = sum(hits) / len(hits) if hits else 0.0
    return {"acc": acc, "f1": acc, "n": len(hits)}

_FLOAT_STR = re.compile(r"^-?\d+(\.\d+)?$")

def score_task(task, outputs, targets, labels=None):
    
    metric = TASKS[task][2]
    if task in NUMERIC_TASKS:
        rec = score_numeric(outputs, targets, **NUMERIC_TASKS[task]); rec["metric"] = "numeric"; return rec
    if not metric.startswith("label_acc"):
        return {"acc": None, "n": len(outputs), "metric": metric}
    if labels is None:
        labels = load_task(task, n=None)[1]
    batched = metric == "label_acc_batched"
    outs, tg = (split_batched(outputs, targets) if batched else (list(outputs), list(targets)))
    is_float = [bool(_FLOAT_STR.match(str(t).strip())) for t in tg]
    if any(is_float):
        lab_o = [o for o, f in zip(outs, is_float) if not f]; lab_t = [t for t, f in zip(tg, is_float) if not f]
        num_o = [o for o, f in zip(outs, is_float) if f];     num_t = [t for t, f in zip(tg, is_float) if f]
        a = score_label_acc(lab_o, lab_t, labels) if lab_t else {"acc": 0.0, "f1": 0.0, "n": 0}
        b = score_numeric(num_o, num_t) if num_t else {"acc": 0.0, "n": 0}
        n = a["n"] + b["n"]
        acc = (a["acc"] * a["n"] + b["acc"] * b["n"]) / n if n else 0.0
        return {"acc": acc, "f1": acc, "n": n, "metric": metric + "+numeric-rows"}
    rec = score_label_acc(outs, tg, labels); rec["metric"] = metric; return rec

def score_label_acc(outputs, targets, labels, batched=False):
    if batched:
        outputs, targets = split_batched(outputs, targets)
    preds = [first_occurrence_pred(o, labels) for o in outputs]
    correct = [int(p is not None and p.lower() == t.lower())
               for p, t in zip(preds, targets)]
    acc = sum(correct) / len(correct) if correct else 0.0
    try:
        from sklearn.metrics import f1_score
        resp = [t if c else str(o) for c, t, o in zip(correct, targets, outputs)]
        f1 = float(f1_score(targets, resp, average="weighted"))
    except Exception:
        f1 = -1.0
    return {"acc": acc, "f1": f1, "n": len(correct)}
