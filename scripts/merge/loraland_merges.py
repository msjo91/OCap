import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res

from src.merge import (RULES, merged_rank, client_norms, _cells, _delta, mad_tau,
                        dcmerge_smooth_pool)
from scripts.analysis.pool_diagnostic import load

BASE = "mistralai/Mistral-7B-v0.1"
TARGETS = ["q_proj", "v_proj"]
DEFAULT_RULES = "concat,isocts,dcmerge,dcmerge_g50,isocts_neq,isocts_white,capmad_isocts,adaptive_mad"

def load_pool(repos):
    pool = {}
    for rp in repos:
        p, cfg = load(rp)
        assert cfg["base_model_name_or_path"] == BASE and cfg["r"] == 8, rp
        assert sorted(cfg["target_modules"]) == TARGETS, rp
        pool[rp.split("/")[-1]] = p
    cells = set(next(iter(pool.values())))
    assert all(set(p) == cells for p in pool.values()), "cell sets differ"
    return pool

def write_adapter(merged, out_dir, rank, description):
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for (layer, proj), fs in merged.items():
        assert fs["A"].shape[0] == rank and fs["B"].shape[1] == rank, (layer, proj)
        stem = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        tensors[f"{stem}.lora_A.weight"] = torch.tensor(np.ascontiguousarray(fs["A"]), dtype=torch.bfloat16).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.tensor(np.ascontiguousarray(fs["B"]), dtype=torch.bfloat16).contiguous()
    save_file(tensors, os.path.join(out_dir, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": BASE,
           "r": rank, "lora_alpha": rank, "lora_dropout": 0.0, "target_modules": TARGETS,
           "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
           "init_lora_weights": True, "use_rslora": False, "use_dora": False}
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=1)
    open(os.path.join(out_dir, "MERGE_INFO.txt"), "w").write(description + "\n")

ISOCTS_FAMILY = {"isocts", "isocts_g50", "isocts_neq", "isocts_white", "capmad_isocts", "adaptive_mad"}

def rule_kwargs(rule, n_r, kc=None):
    fn, kw = RULES[rule]
    if "rank" in kw and kw["rank"] < n_r:
        kw = {**kw, "rank": n_r}
    if rule == "isocts":
        kw = {**kw, "s": None}
    if kc is not None and rule in ISOCTS_FAMILY:
        kw = {**kw, "k_common": n_r // 2 if kc == "half" else int(kc)}
    return fn, kw

def recon_error(pool, merged, cell):
    exact = np.mean([pool[t][cell]["B"] @ pool[t][cell]["A"] for t in pool], axis=0)
    got = merged[cell]["B"] @ merged[cell]["A"]
    return float(np.linalg.norm(got - exact) / np.linalg.norm(exact))

def effective_rank(merged, tol=1e-8):
    cell = sorted(merged)[0]
    A = merged[cell]["A"]
    return int((np.linalg.norm(A, axis=1) > tol * np.linalg.norm(A)).sum())

def cap_binding(pool, k=3.0):
    per_cell, counts = {}, {t: 0 for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        ceil = mad_tau(norms.values(), k) * float(np.median(list(norms.values())))
        bound = sorted(t for t in pool if norms[t] > ceil)
        per_cell[f"{cell[0]}.{cell[1]}"] = {"tau": mad_tau(norms.values(), k), "ceiling": ceil,
                                            "bound": bound,
                                            "scale": {t: min(1.0, ceil / max(norms[t], 1e-12)) for t in bound}}
        for t in bound:
            counts[t] += 1
    n_bound = [len(v["bound"]) for v in per_cell.values()]
    return {"median_bound_per_cell": float(np.median(n_bound)), "mean_bound_per_cell": float(np.mean(n_bound)),
            "min_bound_per_cell": int(min(n_bound)), "max_bound_per_cell": int(max(n_bound)),
            "cells_bound_on_by_client": {t: c for t, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c},
            "n_cells": len(per_cell), "per_cell": per_cell}

def pool_stats(pool):
    cells = _cells(pool)
    per = {t: [float(np.linalg.norm(_delta(pool[t][c], np.float64))) for c in cells] for t in pool}
    mean_norm = {t: float(np.mean(v)) for t, v in per.items()}
    median_norm = {t: float(np.median(v)) for t, v in per.items()}
    cell_spread = []
    for i in range(len(cells)):
        v = [per[t][i] for t in pool]
        cell_spread.append(max(v) / max(min(v), 1e-12))
    med_of_means = float(np.median(list(mean_norm.values())))
    order = sorted(mean_norm, key=mean_norm.get, reverse=True)
    smoothed = dcmerge_smooth_pool(pool, "mean")
    return {
        "n_clients": len(pool), "n_cells": len(cells), "r": 8, "base": BASE,
        "client_mean_norm": {t: mean_norm[t] for t in order},
        "client_median_norm": {t: median_norm[t] for t in order},
        "client_mean_norm_over_pool_median": {t: mean_norm[t] / med_of_means for t in order},
        "spread_of_client_mean_norms": max(mean_norm.values()) / min(mean_norm.values()),
        "spread_of_client_median_norms": max(median_norm.values()) / min(median_norm.values()),
        "per_cell_spread_mean": float(np.mean(cell_spread)),
        "per_cell_spread_median": float(np.median(cell_spread)),
        "per_cell_spread_min": float(np.min(cell_spread)),
        "per_cell_spread_max": float(np.max(cell_spread)),
        "adaptive_mad_cap": {"applied_to": "dcmerge_smooth_pool(pool, 'mean') norms, k=3 (as in merge_adaptive)",
                             **cap_binding(smoothed)},
        "capmad_isocts_cap": {"applied_to": "raw client norms, k=3 (as in merge_capped_raw)",
                              **cap_binding(pool)},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--pool_json", default=res("d1", "predibase_mistral7b.json"))
    ap.add_argument("--out", default=res("loraland_merges"))
    ap.add_argument("--drop", default="")
    ap.add_argument("--kc", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--stats", default="pool_stats.json")
    args = ap.parse_args()
    drop = set(filter(None, args.drop.split(",")))
    repos = [r for r in json.load(open(args.pool_json))["repos"] if r.split("/")[-1] not in drop]
    t0 = time.time()
    pool = load_pool(repos)
    n_r = len(pool) * 8
    cell0 = _cells(pool)[0]
    print(f"loaded {len(pool)} clients x {len(_cells(pool))} cells in {time.time() - t0:.1f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    stats = pool_stats(pool)
    stats["rules"] = {}
    for rule in args.rules.split(","):
        fn, kw = rule_kwargs(rule, n_r, args.kc)
        t1 = time.time()
        merged = fn(pool, **kw)
        for cell, fs in merged.items():
            assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), (rule, cell)
            assert np.abs(fs["B"]).max() > 0, (rule, cell)
        assert set(merged) == set(_cells(pool)), rule
        rank = merged_rank(merged)
        err = recon_error(pool, merged, cell0) if rule == "concat" else None
        eff = effective_rank(merged)
        desc = f"rule={rule} kwargs={kw} pool=loraland n={len(pool)} clients={sorted(pool)}"
        write_adapter(merged, os.path.join(args.out, rule + args.suffix), rank, desc)
        stats["rules"][rule] = {"kwargs": {k: v for k, v in kw.items()}, "rank": rank,
                                "nonzero_directions_cell0": eff, "concat_recon_err_cell0": err}
        print(f"{rule:14s} rank={rank:4d} nonzero={eff:4d} kwargs={kw} "
              f"{'recon_err=%.2e ' % err if err is not None else ''}{time.time() - t1:.1f}s", flush=True)
    json.dump(stats, open(os.path.join(args.out, args.stats), "w"), indent=1)
    print("->", os.path.join(args.out, args.stats))

if __name__ == "__main__":
    main()
