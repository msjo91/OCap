import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

BASE = "mistralai/Mistral-7B-v0.1"
OUT = res("loraland")
MERGE_ROOT = res("loraland_merges")
N_ITEMS = 200
SEED = 42
CARD = "Hugging Face model card 'Sample input' of predibase/{client} (LoRA Land, arXiv 2405.00732)"

def _glue(cfg, split):
    from datasets import load_dataset
    return load_dataset("nyu-mll/glue", cfg, split=split)

def _hf(repo, split):
    from datasets import load_dataset
    return load_dataset(repo, split=split)

def _hellaswag(split):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Rowan/hellaswag", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(p).to_dict("records")

TASKS = {
    "glue_sst2": {
        "client": "glue_sst2", "load": lambda: _glue("sst2", "validation"), "split": "nyu-mll/glue sst2 validation",
        "prompt": lambda x: ("Given the following sentence:\n\n" + x["sentence"] + "\n\nRespond with 0 if the sentiment of "
                             "the sentence is negative and 1 if the sentiment of the sentence is positive."),
        "gold": lambda x: str(x["label"]), "labels": ["0", "1"], "max_new_tokens": 6},
    "glue_mrpc": {
        "client": "glue_mrpc", "load": lambda: _glue("mrpc", "validation"), "split": "nyu-mll/glue mrpc validation",
        "prompt": lambda x: ("You are given two sentences below, Sentence 1 and Sentence 2. If the two sentences are "
                             "semantically equivalent, please return 1. Otherwise, please return 0.\n\n### Sentence 1: "
                             + x["sentence1"] + "\n\n### Sentence 2: " + x["sentence2"] + "\n\n### Label: "),
        "gold": lambda x: str(x["label"]), "labels": ["0", "1"], "max_new_tokens": 6},
    "glue_qnli": {
        "client": "glue_qnli", "load": lambda: _glue("qnli", "validation"), "split": "nyu-mll/glue qnli validation",
        "prompt": lambda x: ("You are provided a question and a corresponding response below. If the response properly "
                             "answers the question, please return 0. Otherwise, please return 1.\n\n### Question: "
                             + x["question"] + "\n\n### Response: " + x["sentence"] + "\n\n### Label: "),
        "gold": lambda x: str(x["label"]), "labels": ["0", "1"], "max_new_tokens": 6},
    "glue_cola": {
        "client": "glue_cola", "load": lambda: _glue("cola", "validation"), "split": "nyu-mll/glue cola validation",
        "prompt": lambda x: ("Determine if the sentence below is syntactically and semantically correct. If it is "
                             "syntactically and semantically correct, respond \"1\". Otherwise, respond \"0\".\n\n"
                             "Sentence: " + x["sentence"] + "\n\nLabel: "),
        "gold": lambda x: str(x["label"]), "labels": ["0", "1"], "max_new_tokens": 6},
    "glue_mnli": {
        "client": "glue_mnli", "load": lambda: _glue("mnli", "validation_matched"),
        "split": "nyu-mll/glue mnli validation_matched",
        "prompt": lambda x: ("You are given a premise and a hypothesis below. If the premise entails the hypothesis, "
                             "return 0. If the premise contradicts the hypothesis, return 2. Otherwise, if the premise "
                             "does neither, return 1.\n\n### Premise: " + x["premise"] + "\n\n### Hypothesis: "
                             + x["hypothesis"] + "\n\n### Label: "),
        "gold": lambda x: str(x["label"]), "labels": ["0", "1", "2"], "max_new_tokens": 6},
    "glue_qqp": {
        "client": "glue_qqp", "load": lambda: _glue("qqp", "validation"), "split": "nyu-mll/glue qqp validation",
        "prompt": lambda x: ("You are given two questions below, Question 1 and Question 2. If the two questions are "
                             "semantically equivalent, please return 1. Otherwise, please return 0.\n\n### Question 1: "
                             + x["question1"] + "\n\n### Question 2: " + x["question2"] + "\n\n### Label: "),
        "gold": lambda x: str(x["label"]), "labels": ["0", "1"], "max_new_tokens": 6},
    "dbpedia": {
        "client": "dbpedia", "load": lambda: _hf("fancyzhx/dbpedia_14", "test"), "split": "fancyzhx/dbpedia_14 test",
        "prompt": lambda x: ("You are given the title and the body of an article below. Please determine the type of "
                             "the article.\n### Title: " + x["title"].strip() + "\n\n### Body: " + x["content"].strip()
                             + "\n\n### Article Type: "),
        "gold": lambda x: str(x["label"]), "labels": [str(i) for i in range(14)], "max_new_tokens": 6},
    "agnews_explained": {
        "client": "agnews_explained", "load": lambda: _hf("fancyzhx/ag_news", "test"), "split": "fancyzhx/ag_news test",
        "prompt": lambda x: ("Below is a news article. Please classify it under one of the following classes (World, "
                             "Business, Sports, Sci/Tech) and provide a reasonable coherent explanation for why the "
                             "article is classified as such. Please format your response as a JSON payload.\n\n"
                             "### Article: " + x["text"] + "\n\n### JSON Response"),
        "gold": lambda x: ["world", "sports", "business", "sci/tech"][x["label"]],
        "labels": ["world", "sports", "business", "sci/tech"], "max_new_tokens": 32, "json_key": "text_label"},
    "hellaswag": {
        "client": "hellaswag", "load": lambda: _hellaswag("validation"), "split": "Rowan/hellaswag validation",
        "prompt": lambda x: ("You are provided with an incomplete passage below as well as 4 endings in quotes and "
                             "separated by commas, with only one of them being the correct ending. Treat the endings as "
                             "being labelled 0, 1, 2, 3 in order. Please respond with the number corresponding to the "
                             "correct ending for the passage.\n\n### Passage: " + x["ctx"] + "\n\n### Endings: "
                             + str(np.array(list(x["endings"]))) + "\n\n### Correct Ending Number: "),
        "gold": lambda x: str(int(x["label"])), "labels": ["0", "1", "2", "3"], "max_new_tokens": 6},
}

def normalize(text, task, lenient=False):
    spec = TASKS[task]
    s = text or ""
    if spec.get("json_key"):
        m = re.search(r'"%s"\s*:\s*"([^"]*)"' % spec["json_key"], s)
        if m:
            s = m.group(1)
    if lenient:
        s = s.lstrip(string.punctuation + string.whitespace)
    s = s.strip().split("\n")[0].strip().lower()
    toks = s.split()
    s = toks[0] if toks else ""
    s = s.strip(string.punctuation + string.whitespace)
    try:
        f = float(s)
        if f == int(f):
            s = str(int(f))
    except ValueError:
        pass
    return s

def items(task):
    spec = TASKS[task]
    ds = spec["load"]()
    n = len(ds)
    idx = sorted(int(i) for i in np.random.default_rng(SEED).choice(n, size=min(N_ITEMS, n), replace=False))
    return [(i, ds[i]) for i in idx]

def task_info(task, its):
    golds = [TASKS[task]["gold"](x) for _, x in its]
    maj = Counter(golds).most_common(1)[0]
    return {"client": TASKS[task]["client"], "split": TASKS[task]["split"], "n": len(its), "seed": SEED,
            "item_indices": [i for i, _ in its], "labels": TASKS[task]["labels"],
            "majority_label": maj[0], "majority_rate": maj[1] / len(golds),
            "uniform_chance": 1.0 / len(TASKS[task]["labels"]),
            "template_source": CARD.format(client=TASKS[task]["client"]),
            "example_prompt": TASKS[task]["prompt"](its[0][1]),
            "max_new_tokens": TASKS[task]["max_new_tokens"]}

def load_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                 device_map={"": 0}, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.unk_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model.generation_config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok

def adapter_path(name):
    if name == "base":
        return None
    if name.startswith("home_"):
        from huggingface_hub import snapshot_download
        return snapshot_download("predibase/" + name[len("home_"):],
                                 allow_patterns=["adapter_model.safetensors", "adapter_config.json"])
    p = os.path.join(MERGE_ROOT, name)
    assert os.path.exists(os.path.join(p, "adapter_model.safetensors")), p
    return p

def run_task(model, tok, task, its, batch_size):
    from src.runner import generate
    spec = TASKS[task]
    prompts = [spec["prompt"](x) for _, x in its]
    lens = [len(t) for t in tok(prompts)["input_ids"]]
    order = np.argsort(lens)[::-1]
    outs_sorted = generate(model, tok, [prompts[i] for i in order], spec["max_new_tokens"],
                           batch_size=batch_size, max_length=4096)
    outs = [None] * len(prompts)
    for k, i in enumerate(order):
        outs[i] = outs_sorted[k]
    rows = []
    for (i, x), p, o in zip(its, prompts, outs):
        g = spec["gold"](x)
        pred = normalize(o, task)
        rows.append({"idx": i, "gold": g, "pred": pred, "correct": pred == g,
                     "valid_label": pred in spec["labels"], "generation": o, "prompt_tokens": len(tok(p)["input_ids"])})
    return rows

def write_metrics(model_name, adapter, tasks):
    res = {}
    for t in tasks:
        f = os.path.join(OUT, f"{model_name}__{t}.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        len_pred = [normalize(r["generation"], t, lenient=True) for r in rows]
        res[t] = {"acc": float(np.mean([r["correct"] for r in rows])), "n": len(rows),
                  "valid_label_rate": float(np.mean([r["valid_label"] for r in rows])),
                  "acc_lenient": float(np.mean([p == r["gold"] for p, r in zip(len_pred, rows)])),
                  "valid_label_rate_lenient": float(np.mean([p in TASKS[t]["labels"] for p in len_pred]))}
    mf = os.path.join(OUT, f"{model_name}__metrics.json")
    merged_tasks = res
    out = {"model": model_name, "adapter": adapter, "base": BASE, "n_per_task": N_ITEMS, "seed": SEED,
           "decoding": "greedy", "quant": "bnb 8-bit", "tasks": merged_tasks,
           "mean": float(np.mean([v["acc"] for v in merged_tasks.values()])) if merged_tasks else None,
           "mean_lenient": float(np.mean([v["acc_lenient"] for v in merged_tasks.values()])) if merged_tasks else None,
           "lenient_scoring": "leading punctuation/whitespace stripped before taking the first line",
           "mean_over": sorted(merged_tasks)}
    json.dump(out, open(mf, "w"), indent=1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--home_only", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tasks = args.tasks.split(",")
    data = {t: items(t) for t in tasks}
    info_f = os.path.join(OUT, "tasks.json")
    info = json.load(open(info_f)) if os.path.exists(info_f) else {}
    for t in tasks:
        info[t] = task_info(t, data[t])
    json.dump(info, open(info_f, "w"), indent=1)
    t0 = time.time()
    base, tok = load_base()
    print(f"base loaded in {time.time() - t0:.0f}s", flush=True)
    from peft import PeftModel
    peft_model, prev = None, None
    for name in args.models.split(","):
        path = adapter_path(name)
        my_tasks = [t for t in tasks if not args.home_only or name in ("base", "home_" + TASKS[t]["client"])]
        if args.skip_done:
            my_tasks = [t for t in my_tasks if not os.path.exists(os.path.join(OUT, f"{name}__{t}.jsonl"))]
        if not my_tasks:
            print(f"{name}: nothing to do", flush=True)
            continue
        if path is None:
            model = peft_model if peft_model is not None else base
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base, path, adapter_name=name)
            else:
                peft_model.load_adapter(path, adapter_name=name)
            peft_model.set_adapter(name)
            if prev is not None:
                peft_model.delete_adapter(prev)
            prev = name
            model = peft_model
            model.eval()
        for t in my_tasks:
            t1 = time.time()
            if path is None and peft_model is not None:
                with peft_model.disable_adapter():
                    rows = run_task(model, tok, t, data[t], args.batch_size)
            else:
                rows = run_task(model, tok, t, data[t], args.batch_size)
            with open(os.path.join(OUT, f"{name}__{t}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            acc = np.mean([r["correct"] for r in rows])
            valid = np.mean([r["valid_label"] for r in rows])
            print(f"{name:28s} {t:18s} acc={acc:.3f} valid={valid:.3f} {time.time() - t1:.0f}s", flush=True)
        m = write_metrics(name, path, my_tasks if args.home_only and name != "base" else tasks)
        print(f"{name:28s} mean={m['mean']:.4f} over {m['mean_over']}", flush=True)

if __name__ == "__main__":
    main()
