import json
import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from . import tasks as T
from .pool import BASE_MODEL
from .stamps import (GEN_CFG, TASKS_AFFECTED_BY_V2, SCORER_VERSION, heldout_gen_cfg,
                     generation_is_current, pool_gen_cfg)

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

def load_model(adapter=None, base=BASE_MODEL, quant_bits=8):
    
    bnb = BitsAndBytesConfig(load_in_8bit=quant_bits == 8, load_in_4bit=quant_bits == 4,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if len(tok) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tok))
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok

def apply_prompt_style(tok, contexts, prompt_style):
    
    if prompt_style in (None, "raw"):
        return list(contexts)
    if prompt_style != "chat":
        raise ValueError(f"unknown prompt_style {prompt_style!r}")
    if getattr(tok, "chat_template", None) is None:
        raise ValueError("tokenizer has no chat template; --prompt_style chat is not available")
    out = []
    for c in contexts:
        p = tok.apply_chat_template([{"role": "user", "content": c}],
                                    tokenize=False, add_generation_prompt=True)
        if tok.bos_token and p.startswith(tok.bos_token):
            p = p[len(tok.bos_token):]
        out.append(p)
    return out

def generate(model, tok, prompts, max_new_tokens, batch_size=8, max_length=8192):
    
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        lens = [len(x) for x in tok(batch, truncation=True, max_length=max_length)["input_ids"]]
        sub = batch_size if max(lens) <= 2048 else (4 if max(lens) <= 4096 else (2 if max(lens) <= 6144 else 1))
        for j in range(0, len(batch), sub):
            outs += _gen_batch(model, tok, batch[j:j + sub], max_new_tokens, max_length)
    return outs

def _gen_batch(model, tok, batch, max_new_tokens, max_length):
    
    outs = []
    if True:
        tokens = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length, return_token_type_ids=False).to(model.device)
        oom = False
        try:
            with torch.no_grad():
                res = model.generate(**tokens, max_new_tokens=max_new_tokens,
                                     do_sample=False, eos_token_id=tok.eos_token_id)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if len(batch) == 1 or not ("out of memory" in msg or "cublaslt ran into an error" in msg
                                       or isinstance(e, torch.cuda.OutOfMemoryError)):
                raise
            print(f"  recoverable error on batch of {len(batch)}: {str(e)[:120]}", flush=True)
            try:
                torch.cuda.synchronize()
            except RuntimeError as e2:
                raise RuntimeError("device fault during generation") from e
            oom = True
        if oom:
            del tokens
            import gc; gc.collect(); torch.cuda.empty_cache()
            print(f"  OOM on batch of {len(batch)} (max_length={max_length}); retrying one at a time", flush=True)
            for x in batch:
                outs += _gen_batch(model, tok, [x], max_new_tokens, max_length)
            return outs
        new_tokens = res[:, tokens["input_ids"].shape[1]:]
        outs += [tok.decode(row, skip_special_tokens=True).strip()
                 for row in new_tokens]
    return outs

def generate_continuation(model, tok, prompts, max_new_tokens, batch_size, max_length):
    
    return generate(model, tok, prompts, max_new_tokens, batch_size=batch_size, max_length=max_length)

def _adapter_mtime(adapter):
    
    if adapter and os.path.isdir(adapter):
        f = os.path.join(adapter, "adapter_model.safetensors")
        if not os.path.exists(f):
            raise FileNotFoundError(f"adapter dir without weights: {adapter}")
        return round(os.path.getmtime(f), 1)
    return None

def _write_metrics(out_dir, model_name, results, task=None):
    
    fn = os.path.join(out_dir, f"{model_name}__metrics.json")
    on_disk = json.load(open(fn)) if os.path.exists(fn) else {}
    if task is not None:
        on_disk[task] = results[task]
        results.clear(); results.update(on_disk)
    else:
        on_disk.update(results); results.clear(); results.update(on_disk)
    tmp = fn + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, fn)

def _pending(model_name, task_names, out_dir, n, force, adapter=None, stamp=None, gen_cfg_for=None,
             suite=None):
    
    metrics_fn = os.path.join(out_dir, f"{model_name}__metrics.json")
    results = json.load(open(metrics_fn)) if os.path.exists(metrics_fn) else {}
    if force:
        return results, list(task_names)
    todo, skipped = [], []
    for name in task_names:
        rec = results.get(name)
        want = gen_cfg_for(name) if gen_cfg_for else GEN_CFG
        done = (rec is not None
                and rec.get("n_requested", n) == n
                and rec.get("adapter_mtime", stamp) == stamp
                and generation_is_current(rec, name, want,
                                          suite or ("heldout" if gen_cfg_for else "pool")))
        (skipped if done else todo).append(name)
    if skipped:
        print(f"[{model_name}] resume: skipping done tasks {skipped}", flush=True)
    return results, todo

def eval_model_on_heldout(model_name, adapter, task_names, n, out_dir,
                          seed=42, quant_bits=8, force=False, clean=False, prompt_style="raw"):
    from . import heldout as H
    os.makedirs(out_dir, exist_ok=True)
    stamp = _adapter_mtime(adapter)
    results, task_names = _pending(model_name, task_names, out_dir, n, force, adapter, stamp,
                                   gen_cfg_for=lambda t: heldout_gen_cfg(H.TASKS[t], t, prompt_style))
    if not task_names:
        print(f"[{model_name}] nothing to do", flush=True)
        return results
    model, tok = load_model(adapter, quant_bits=quant_bits)
    for name in task_names:
        t0 = time.time()
        cfg = H.TASKS[name]
        rows = H.load_task(name, n=n, seed=seed, clean=clean)
        outs = generate_continuation(model, tok,
                                     apply_prompt_style(tok, [r["context"] for r in rows], prompt_style),
                                     cfg["max_new_tokens"], cfg["batch_size"],
                                     cfg["max_length"])
        with open(os.path.join(out_dir, f"{model_name}__{name}.jsonl"), "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "target": r["target"],
                                    "output": o}) + "\n")
        rec = {"model": model_name, "task": name, "metric": cfg["metric"], "clean": clean,
               "n_requested": n, "seconds": round(time.time() - t0, 1),
               "adapter_mtime": stamp, "prompt_style": prompt_style,
               "gen_cfg": heldout_gen_cfg(cfg, name, prompt_style)}
        rec.update(H.score(name, outs, [r["target"] for r in rows]))
        rec["scorer"] = SCORER_VERSION
        results[name] = rec
        _write_metrics(out_dir, model_name, results, task=name)
        acc = rec.get("acc")
        print(f"[{model_name}] {name}: "
              f"{'acc=%.3f' % acc if acc is not None else 'gens saved (rouge offline)'} "
              f"n={rec['n']} {rec['seconds']}s", flush=True)
    return results

def eval_model_on_tasks(model_name, adapter, task_names, n, out_dir,
                        seed=42, batch_size=8, quant_bits=8, force=False, clean=False,
                        prompt_style="raw"):
    
    os.makedirs(out_dir, exist_ok=True)
    stamp = _adapter_mtime(adapter)
    results, task_names = _pending(model_name, task_names, out_dir, n, force, adapter, stamp,
                                   gen_cfg_for=(None if prompt_style in (None, "raw")
                                                else (lambda t: pool_gen_cfg(prompt_style))),
                                   suite="pool")
    if not task_names:
        print(f"[{model_name}] nothing to do", flush=True)
        return results
    model, tok = load_model(adapter, quant_bits=quant_bits)
    for name in task_names:
        t0 = time.time()
        rows, labels = T.load_task(name, n=n, seed=seed, clean=clean)
        _, max_new, metric, owner = T.TASKS[name]
        bs = 4 if (name.startswith("xbrl") or name in ("financebench", "formula")) \
            else batch_size
        outs = generate(model, tok,
                        apply_prompt_style(tok, [r["context"] for r in rows], prompt_style),
                        max_new, batch_size=bs)
        gen_fn = os.path.join(out_dir, f"{model_name}__{name}.jsonl")
        with open(gen_fn, "w") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({"idx": r["_idx"], "target": r["target"],
                                    "output": o}) + "\n")
        rec = {"model": model_name, "task": name, "metric": metric,
               "n": len(rows), "n_requested": n, "clean": clean,
               "seconds": round(time.time() - t0, 1),
               "adapter_mtime": stamp, "prompt_style": prompt_style,
               "gen_cfg": pool_gen_cfg(prompt_style)}
        if metric.startswith("label_acc"):
            rec.update(T.score_task(name, outs, [r["target"] for r in rows], labels))
            rec["scorer"] = SCORER_VERSION
        results[name] = rec
        _write_metrics(out_dir, model_name, results, task=name)
        acc = rec.get("acc")
        print(f"[{model_name}] {name}: "
              f"{'acc=%.3f f1=%.3f' % (rec['acc'], rec['f1']) if acc is not None else 'gens saved (bertscore offline)'} "
              f"n={rec['n']} {rec['seconds']}s", flush=True)
    return results
