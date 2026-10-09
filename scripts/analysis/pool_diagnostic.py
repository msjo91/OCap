import argparse, json, os, re, sys
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.merge import _orthobasis, _mean_cos, chance_frame_similarity

KEY = re.compile(r"layers\.(\d+)\.(?:self_attn|attn)\.([a-z_0-9]+)\.lora_(A|B)\.weight")

def load(repo):
    
    import torch
    from safetensors.torch import load_file
    path = snapshot_download(repo, allow_patterns=["adapter_model.safetensors",
                                                   "adapter_model.bin", "adapter_config.json"])
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    if cfg.get("use_dora"):
        raise ValueError(f"{repo}: DoRA - B@A is not the weight delta")
    r = cfg["r"]
    scale = cfg["lora_alpha"] / ((r ** 0.5) if cfg.get("use_rslora") else r)
    st = os.path.join(path, "adapter_model.safetensors")
    w = load_file(st) if os.path.exists(st) else torch.load(
        os.path.join(path, "adapter_model.bin"), map_location="cpu", weights_only=True)
    out = {}
    for k, t in w.items():
        m = KEY.search(k)
        if not m:
            continue
        cell = (int(m.group(1)), m.group(2))
        out.setdefault(cell, {})[m.group(3)] = t.to(torch.float64).numpy()
    for cell, fs in out.items():
        if "B" in fs:
            fs["B"] = fs["B"] * scale
    return {c: fs for c, fs in out.items() if "A" in fs and "B" in fs}, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repos = args.repos.split(",") if args.repos else [m.id for m in HfApi().list_models(author=args.author, limit=args.limit)]

    pools, cfgs, skipped = {}, {}, []
    for rp in repos:
        try:
            p, c = load(rp)
            if p:
                pools[rp], cfgs[rp] = p, c
            else:
                skipped.append((rp, "no attention LoRA cells"))
        except Exception as e:
            skipped.append((rp, f"{type(e).__name__}: {str(e)[:60]}"))
    print(f"loaded {len(pools)} adapters, skipped {len(skipped)}")
    for s in skipped[:8]:
        print("   skip", s)
    if len(pools) < 2:
        print("need >= 2 adapters"); return

    from collections import defaultdict
    groups = defaultdict(list)
    for rp, c in cfgs.items():
        groups[(c["base_model_name_or_path"], c["r"])].append(rp)
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"   group base={key[0]} r={key[1]}: {len(members)} adapters")
    key, members = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(members) < 2:
        print("no homogeneous group with >= 2 adapters"); return
    pools = {rp: pools[rp] for rp in members}
    print(f"\nANALYSING: base={key[0]}  r={key[1]}  n_clients={len(pools)}")
    cells = sorted(set.intersection(*[set(p) for p in pools.values()]))
    print(f"common cells: {len(cells)}")

    norms = {}
    for rp, p in pools.items():
        norms[rp] = float(np.median([np.linalg.norm(p[c]["B"] @ p[c]["A"]) for c in cells]))
    order = sorted(norms, key=norms.get)
    spread = norms[order[-1]] / max(norms[order[0]], 1e-12)
    print(f"\n||dW|| spread = {spread:.1f}x   "
          f"min {order[0].split('/')[-1]} {norms[order[0]]:.4f}   max {order[-1].split('/')[-1]} {norms[order[-1]]:.4f}")
    for rp in order:
        print(f"   {rp.split('/')[-1]:28s} {norms[rp]:.4f}")

    projs = sorted({c[1] for c in cells})
    summary = {}
    for proj in projs:
        pc = [c for c in cells if c[1] == proj]
        if not pc:
            continue
        names = list(pools)
        simA, simB = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = [_mean_cos(_orthobasis(pools[names[i]][c]["A"].T), _orthobasis(pools[names[j]][c]["A"].T)) for c in pc]
                b = [_mean_cos(_orthobasis(pools[names[i]][c]["B"]),   _orthobasis(pools[names[j]][c]["B"]))   for c in pc]
                simA.append(np.mean(a)); simB.append(np.mean(b))
        d_in = pools[names[0]][pc[0]]["A"].shape[1]; d_out = pools[names[0]][pc[0]]["B"].shape[0]
        r = pools[names[0]][pc[0]]["A"].shape[0]
        chA_m, chA_s = chance_frame_similarity(d_in, r)
        chB_m, chB_s = chance_frame_similarity(d_out, r)
        summary[proj] = {"sim_A": float(np.mean(simA)), "sim_B": float(np.mean(simB)),
                         "chance_A": float(chA_m), "chance_B": float(chB_m),
                         "excess_A_sigma": float((np.mean(simA) - chA_m) / chA_s),
                         "excess_B_sigma": float((np.mean(simB) - chB_m) / chB_s),
                         "pairs": len(simA), "cells": len(pc)}
        s = summary[proj]
        print(f"  {proj:10s} A {s['sim_A']:.3f} (chance {s['chance_A']:.3f}, {s['excess_A_sigma']:+.1f} sigma) | "
              f"B {s['sim_B']:.3f} (chance {s['chance_B']:.3f}, {s['excess_B_sigma']:+.1f} sigma)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"repos": list(pools), "base": key[0], "r": key[1], "n_clients": len(pools),
               "norms": norms, "spread": spread, "summary": summary, "skipped": skipped},
              open(args.out, "w"), indent=1)
    print("\n->", args.out)

if __name__ == "__main__":
    main()
