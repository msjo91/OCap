import argparse, csv, hashlib, json, os, re, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import ROOT
csv.field_size_limit(10 ** 9)
from src import tasks as T, heldout as H

TRAIN = os.path.join(ROOT, "external", "FinLoRA", "data", "train")
XBRL_CSV = os.path.join(TRAIN, "xbrl_csv")
TRAIN_FOR = {
    "sentiment": ["finlora_sentiment_train.jsonl"], "headline": ["headline_train.jsonl"],
    "ner": ["ner_train.jsonl"], "finer": ["finer_train_batched.jsonl"],
    "xbrl_term": ["xbrl_term_train.jsonl"], "formula": ["formula_train.jsonl"],
    "financebench": ["financebench_train.jsonl"], "xbrl_extract": ["xbrl_csv"],
}

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s).lower()).strip())

def h(s):
    return hashlib.md5(norm(s).encode()).hexdigest()

def shingles(s, k=8):
    w = norm(s).split()
    return {hashlib.md5(" ".join(w[i:i + k]).encode()).hexdigest()[:10] for i in range(max(0, len(w) - k + 1))}

def load_train(owner, cap):
    
    rows = []
    for f in TRAIN_FOR.get(owner, []):
        p = os.path.join(TRAIN, f)
        if f == "xbrl_csv":
            if not os.path.isdir(XBRL_CSV):
                continue
            for c in sorted(os.listdir(XBRL_CSV)):
                if not c.endswith(".csv"):
                    continue
                with open(os.path.join(XBRL_CSV, c), newline="") as fh:
                    for i, r in enumerate(csv.DictReader(fh)):
                        if len(rows) >= cap:
                            break
                        rows.append((r.get("input") or "", r.get("output") or ""))
            continue
        if not os.path.exists(p):
            continue
        for line in open(p):
            if len(rows) >= cap:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append((r.get("context", ""), str(r.get("target", ""))))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--near", action="store_true")
    ap.add_argument("--max_train", type=int, default=40000)
    args = ap.parse_args()

    print("POOL TASKS — test split is FinLoRA's own data/test/*.jsonl, distinct files from data/train/")
    hdr = ("task", "owner", "test n", "scored", "%", "train n", "dup ctx", "dup pair", "test dups", "near")
    print("".join(f"{x:>13s}" for x in hdr))
    for t, (f, _, metric, owner) in T.TASKS.items():
        full, _ = T.load_task(t, n=None)
        scored = min(args.n, len(full))
        tr = load_train(owner, args.max_train)
        tr_ctx = {h(c) for c, _ in tr}
        tr_pair = {h(c + "||" + g) for c, g in tr}
        sub, _ = T.load_task(t, n=scored, seed=42)
        d_ctx = sum(1 for r in sub if h(r["context"]) in tr_ctx)
        d_pair = sum(1 for r in sub if h(r["context"] + "||" + str(r["target"])) in tr_pair)
        seen = collections.Counter(h(r["context"]) for r in full)
        test_dups = sum(v - 1 for v in seen.values() if v > 1)
        near = "-"
        if args.near and tr:
            tr_sh = [shingles(c) for c, _ in tr[:3000]]
            hit = 0
            for r in sub[:300]:
                s = shingles(r["context"])
                if s and max((len(s & x) / max(len(s | x), 1) for x in tr_sh), default=0) >= 0.8:
                    hit += 1
            near = f"{hit}/300"
        print(f"{t[:13]:>13s}{owner[:13]:>13s}{len(full):>13d}{scored:>13d}"
              f"{100*scored/len(full):>12.0f}%{len(tr):>13d}{d_ctx:>13d}{d_pair:>13d}{test_dups:>13d}{near:>13s}")

    print("\nHELD-OUT TASKS — our own ingests; no client trained on them (see the overlap audit)")
    print("".join(f"{x:>13s}" for x in ("task", "test n", "scored", "%", "test dups")))
    for t in H.TASKS:
        rows = H.load_task(t, n=None)
        scored = min(args.n, len(rows))
        seen = collections.Counter(h(r["context"]) for r in rows)
        print(f"{t[:13]:>13s}{len(rows):>13d}{scored:>13d}{100*scored/len(rows):>12.0f}%"
              f"{sum(v-1 for v in seen.values() if v>1):>13d}")

if __name__ == "__main__":
    main()
