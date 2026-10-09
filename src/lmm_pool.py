import json
import os
import re

import numpy as np
import torch
from safetensors.torch import save_file

BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
KEY_RE = re.compile(r"layers\.(\d+)\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.lora_(A|B)\.weight")
PROJ_RE = re.compile(r"mm_projector\.(0|2)\.(weight|bias)$")
PROJ_MAP = {"0": "linear_1", "2": "linear_2"}
TASKS = ["ScienceQA", "ImageNet", "VQAv2", "REC", "OCRVQA", "VizWiz", "flickr30k", "IconQA"]
TARGET_MODULES_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

def adapter_dir(root, task):
    return os.path.join(root, "adapters", f"LLaVA_7B_lora_r16_{task}")

def load_llava_adapter(path, fold_alpha=True, dtype=np.float64):
    
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    scale = (cfg["lora_alpha"] / cfg["r"]) if fold_alpha else 1.0
    sd = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu", weights_only=True)
    out = {}
    for key, t in sd.items():
        m = KEY_RE.search(key)
        if not m:
            continue
        layer, module, factor = int(m.group(1)), m.group(2), m.group(3)
        arr = t.float().numpy().astype(dtype)
        if factor == "B":
            arr = arr * scale
        out.setdefault((layer, module), {})[factor] = arr
    if not out:
        raise ValueError(f"{path}: no LoRA factors matched {KEY_RE.pattern}")
    proj = {}
    pf = os.path.join(path, "non_lora_trainables.bin")
    if os.path.exists(pf):
        for key, t in torch.load(pf, map_location="cpu", weights_only=True).items():
            m = PROJ_RE.search(key)
            if m:
                proj[f"{PROJ_MAP[m.group(1)]}.{m.group(2)}"] = t.float()
    return out, proj, cfg

def load_llava_pool(root, tasks=None):
    tasks = tasks or TASKS
    factors, projs = {}, {}
    for t in tasks:
        f, p, _ = load_llava_adapter(adapter_dir(root, t))
        factors[t], projs[t] = f, p
    return factors, projs

def mean_projector(projs):
    
    keys = sorted(next(iter(projs.values())))
    return {k: torch.stack([projs[t][k] for t in projs]).mean(0) for k in keys}

def write_llava_merged(factors, out_dir, rank, projector=None, description=""):
    
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for (layer, module), fs in factors.items():
        assert fs["A"].shape[0] == rank and fs["B"].shape[1] == rank, f"rank mismatch at {(layer, module)}"
        stem = f"base_model.model.model.language_model.layers.{layer}.{module}"
        tensors[f"{stem}.lora_A.weight"] = torch.tensor(np.ascontiguousarray(fs["A"]), dtype=torch.float16).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.tensor(np.ascontiguousarray(fs["B"]), dtype=torch.float16).contiguous()
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": None, "base_model_name_or_path": BASE_MODEL,
           "r": rank, "lora_alpha": rank, "lora_dropout": 0.0, "bias": "none",
           "target_modules": TARGET_MODULES_RE,
           "layers_pattern": None, "layers_to_transform": None, "modules_to_save": None,
           "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
           "use_rslora": False, "use_dora": False}
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=1)
    if projector:
        save_file({k: v.contiguous() for k, v in projector.items()}, os.path.join(out_dir, "projector.safetensors"))
    if description:
        open(os.path.join(out_dir, "MERGE_INFO.txt"), "w").write(description + "\n")
    return out_dir

def read_llava_merged(path, dtype=np.float64):
    
    from safetensors.torch import load_file
    out = {}
    for key, t in load_file(os.path.join(path, "adapter_model.safetensors")).items():
        m = KEY_RE.search(key)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t.float().numpy().astype(dtype)
    pf = os.path.join(path, "projector.safetensors")
    proj = load_file(pf) if os.path.exists(pf) else {}
    return out, proj

def projector_module(model):
    
    for name, mod in model.named_modules():
        if name.endswith("multi_modal_projector"):
            return mod
    raise AttributeError("no multi_modal_projector submodule found")

def apply_projector(model, projector):
    
    with torch.no_grad():
        mp = projector_module(model)
        for k, v in projector.items():
            layer, param = k.split(".")
            dst = getattr(getattr(mp, layer), param)
            dst.copy_(v.to(dst.dtype))
    return model
