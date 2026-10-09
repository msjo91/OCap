import json
import os
import re

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

TASKS = ["sentiment", "headline", "ner", "finer", "xbrl_extract",
         "xbrl_term", "formula", "financebench"]
POOL_SUFFIX = {"r8": "8bits_r8", "rslora": "8bits_r8_rslora", "dora": "8bits_r8_dora"}
BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
PROJS = ("q_proj", "k_proj", "v_proj")
N_LAYERS = 32

KEY_RE = re.compile(
    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.lora_(A|B)\.weight")

def adapter_repo(task, pool="r8"):
    return f"wangd12/{task}_llama_3_1_8b_{POOL_SUFFIX[pool]}"

def peft_scale(cfg):
    
    return cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])

def load_adapter_factors(task, pool="r8", fold_alpha=True, dtype=np.float64, allow_dora=False):
    
    path = snapshot_download(adapter_repo(task, pool))
    return load_local_adapter_factors(path, fold_alpha=fold_alpha, dtype=dtype, allow_dora=allow_dora,
                                      label=f"{task}/{pool}")

def load_local_adapter_factors(path, fold_alpha=True, dtype=np.float64, allow_dora=False, label=None):
    
    label = label or path
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    if cfg.get("use_dora"):
        if not allow_dora:
            raise ValueError(f"{label}: DoRA adapter — B@A is not its weight delta (magnitude vector "
                             "dropped). Pass allow_dora=True for subspace diagnostics only; never merge it.")
    scale = peft_scale(cfg) if fold_alpha else 1.0
    out = {}
    with safe_open(os.path.join(path, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            m = KEY_RE.search(key)
            if not m:
                continue
            layer, proj, factor = int(m.group(1)), m.group(2), m.group(3)
            t = f.get_tensor(key).float().numpy().astype(dtype)
            if factor == "B":
                t = t * scale
            out.setdefault((layer, proj), {})[factor] = t
    if not out:
        raise ValueError(f"{label}: no q/k/v LoRA factors found in adapter_model.safetensors")
    return out, cfg

def load_pool(tasks=None, pool="r8", allow_dora=False):
    tasks = tasks or TASKS
    return {t: load_adapter_factors(t, pool, allow_dora=allow_dora)[0] for t in tasks}

def write_merged_adapter(factors, out_dir, rank, description=""):
    
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for (layer, proj), fs in factors.items():
        assert fs["A"].shape[0] == rank and fs["B"].shape[1] == rank, \
            f"rank mismatch at {(layer, proj)}: A{fs['A'].shape} B{fs['B'].shape}"
        stem = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        tensors[f"{stem}.lora_A.weight"] = torch.tensor(
            np.ascontiguousarray(fs["A"]), dtype=torch.bfloat16).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.tensor(
            np.ascontiguousarray(fs["B"]), dtype=torch.bfloat16).contiguous()
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))

    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE_MODEL,
        "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
        "target_modules": list(PROJS), "bias": "none",
        "fan_in_fan_out": False, "inference_mode": True,
        "init_lora_weights": True, "use_rslora": False, "use_dora": False,
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if description:
        with open(os.path.join(out_dir, "MERGE_INFO.txt"), "w") as f:
            f.write(description + "\n")
    return out_dir
