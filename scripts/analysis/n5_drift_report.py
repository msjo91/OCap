import glob, json, os, sys
import numpy as np
from safetensors.numpy import load_file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.pool import KEY_RE
from src.merge import frame_similarity, chance_frame_similarity
from scripts.merge.n5_merges import pool_members, N5_DIR, REGIMES

def load(d):
    out, init = {}, {}
    for k, t in load_file(os.path.join(d, "adapter_model.safetensors")).items():
        m = KEY_RE.search(k)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t.astype(np.float64)
    for k, t in load_file(os.path.join(d, "init_A.safetensors")).items():
        m = KEY_RE.search(k)
        if m:
            init[(int(m.group(1)), m.group(2))] = t.astype(np.float64)
    return out, init

def main():
    print(f"{'regime':8s} {'clients':>7s} {'A drift ||A-A0||/||A0|| (median; per client)':>44s} {'||B||':>7s} {'A-sim within pool':>18s} {'B-sim':>7s}")
    for reg in ("S_fz", "S_lr01", "S", "I"):
        members = [m for m in pool_members(reg) if os.path.exists(os.path.join(N5_DIR, m, "adapter_model.safetensors"))]
        if len(members) < 2:
            print(f"{reg:8s} {len(members):>7d}  (not trained yet)"); continue
        fs = {m: load(os.path.join(N5_DIR, m)) for m in members}
        cells = sorted(next(iter(fs.values()))[0])
        drift = {m: float(np.median([np.linalg.norm(f[c]["A"] - a0[c]) / np.linalg.norm(a0[c]) for c in cells])) for m, (f, a0) in fs.items()}
        nb = float(np.median([np.linalg.norm(f[c]["B"]) for f, _ in fs.values() for c in cells]))
        pool = {m: f for m, (f, _) in fs.items()}
        names, S = frame_similarity(pool, "A"); _, SB = frame_similarity(pool, "B")
        off = [S[i, j] for i in range(len(names)) for j in range(i + 1, len(names))]
        offb = [SB[i, j] for i in range(len(names)) for j in range(i + 1, len(names))]
        r, d_in = pool[names[0]][cells[0]]["A"].shape
        ch, sd = chance_frame_similarity(d_in, r)
        print(f"{reg:8s} {len(members):>7d} {np.median(list(drift.values())):>8.3f}  "
              + " ".join(f"{m.split('_')[0][:4]} {v:.3f}" for m, v in drift.items()).ljust(35)
              + f" {nb:>7.3f} {np.mean(off):>8.3f} (chance {ch:.3f}) {np.mean(offb):>7.3f}")
    print("\nregimes:", "; ".join(f"{k} = {v}" for k, v in REGIMES.items()))
    print("merge outcome per regime: python scripts/analysis/n5_aggregate.py --pools S_fz,S_lr01,S,I")

if __name__ == "__main__":
    main()
