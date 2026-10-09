#!/usr/bin/env python

import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import ROOT

ROOT = Path(ROOT)
EXT = ROOT / "external"
OUT = ROOT / "data/heldout"
FINEXAM_SRC = Path(os.environ.get("FINEXAM_SRC",
                                  EXT / "FinExam10k" / "finexam10k_public_5110_canonical.jsonl"))

def write_jsonl(name, rows):
    p = OUT / name / "test.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: wrote {len(rows)} rows -> {p}")
    return len(rows)

def serialize_table(table):
    return "\n".join(" | ".join(str(c) for c in row) for row in table)

def join_text(x):
    
    if isinstance(x, list):
        return " ".join(s.strip() for s in x if s and s.strip() and s.strip() != ".")
    return str(x).strip()

def doc_context(e):
    parts = []
    pre = join_text(e.get("pre_text", ""))
    if pre:
        parts.append(pre)
    table = e.get("table") or []
    if table:
        parts.append("Table:\n" + serialize_table(table))
    post = join_text(e.get("post_text", ""))
    if post:
        parts.append(post)
    return "\n\n".join(parts)

def ingest_finqa():
    data = json.load(open(EXT / "FinQA/dataset/test.json"))
    rows = []
    for e in data:
        qa = e["qa"]
        ctx = (doc_context(e)
               + "\n\nQuestion: " + qa["question"].strip()
               + "\nAnswer:")
        rows.append({
            "id": e["id"],
            "context": ctx,
            "target": str(qa["exe_ans"]),
            "meta": {
                "program": qa.get("program"),
                "answer": qa.get("answer"),
                "filename": e.get("filename"),
            },
        })
    return write_jsonl("finqa", rows)

def ingest_convfinqa():
    data = json.load(open(EXT / "ConvFinQA/data/dev.json"))
    rows = []
    for e in data:
        ann = e["annotation"]
        questions = ann["dialogue_break"]
        exe_answers = ann["exe_ans_list"]
        programs = ann["turn_program"]
        assert len(questions) == len(exe_answers) == len(programs)
        doc = doc_context(e)
        for i, (q, a, prog) in enumerate(zip(questions, exe_answers, programs)):
            history = ""
            for j in range(i):
                history += f"\nQ: {questions[j].strip()}\nA: {exe_answers[j]}"
            ctx = (doc
                   + ("\n\nConversation so far:" + history if history else "")
                   + f"\n\nQ: {q.strip()}\nA:")
            rows.append({
                "id": f"{e['id']}_turn{i}",
                "context": ctx,
                "target": str(a),
                "meta": {
                    "program": prog,
                    "turn_index": i,
                    "num_turns": len(questions),
                    "conversation_id": e["id"],
                    "filename": e.get("filename"),
                },
            })
    return write_jsonl("convfinqa", rows)

ECT_INSTRUCTION = (
    "Summarize the following earnings-call transcript into concise "
    "bullet-point highlights covering the key reported financial figures "
    "and guidance.\n\nTranscript:\n")

def ingest_ectsum():
    tdir = EXT / "ECTSum/data/final/test"
    ects = sorted((tdir / "ects").iterdir())
    rows = []
    for fp in ects:
        gt = tdir / "gt_summaries" / fp.name
        transcript = fp.read_text().strip()
        summary = gt.read_text().strip()
        rows.append({
            "id": fp.stem,
            "context": ECT_INSTRUCTION + transcript + "\n\nSummary:",
            "target": summary,
            "meta": {"source_file": fp.name},
        })
    return write_jsonl("ectsum", rows)

def ingest_finexam10k():
    rows = []
    with open(FINEXAM_SRC) as f:
        for line in f:
            r = json.loads(line)
            opts = []
            for L in ["A", "B", "C", "D"]:
                v = (r.get(f"option_{L}") or "").strip()
                if v:
                    opts.append(f"{L}. {v}")
            ctx = (r["question"].strip() + "\n\n" + "\n".join(opts)
                   + "\n\nAnswer with the letter only.")
            rows.append({
                "id": r["id"],
                "context": ctx,
                "target": r["answer"].strip(),
                "meta": {
                    "exam": r.get("exam"),
                    "level": r.get("level"),
                    "program_stage": r.get("program_stage"),
                    "difficulty": r.get("difficulty"),
                    "context_complete": r.get("context_complete"),
                    "source_category": r.get("source_category"),
                    "answer_text": r.get("answer_text"),
                },
            })
    return write_jsonl("finexam10k", rows)

if __name__ == "__main__":
    counts = {}
    for name, fn in [("finqa", ingest_finqa), ("convfinqa", ingest_convfinqa),
                     ("ectsum", ingest_ectsum), ("finexam10k", ingest_finexam10k)]:
        try:
            counts[name] = fn()
        except FileNotFoundError as e:
            print(f"{name}: SKIPPED, source not found ({e.filename}); see README 'Data'", file=sys.stderr)
    print(json.dumps(counts))
