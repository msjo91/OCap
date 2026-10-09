import argparse, glob, json, os, sys
import numpy as np
from safetensors.numpy import load_file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src.pool import KEY_RE
from scripts.analysis.d1_subspace_diagnostic import orthobasis, mean_cos, chance_level

def load(d):
    out = {}
    w = load_file(os.path.join(d, "adapter_model.safetensors"))
    for k, t in w.items():
        m = KEY_RE.search(k)
        if m:
            out.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t.astype(np.float64)
    a0 = load_file(os.path.join(d, "init_A.safetensors"))
    init = {}
    for k, t in a0.items():
        m = KEY_RE.search(k)
        if m:
            init[(int(m.group(1)), m.group(2))] = t.astype(np.float64)
    return out, init

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=res("n5")); args = ap.parse_args()
    clients = {}
    for d in sorted(glob.glob(os.path.join(args.dir, "*_S")) + glob.glob(os.path.join(args.dir, "*_I"))
                    + glob.glob(os.path.join(args.dir, "*_I2"))):
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            clients[os.path.basename(d)] = load(d)
    print("clients:", list(clients))
    cells = sorted(next(iter(clients.values()))[0].keys())
    print("\nper-client factor stats (median over cells; random-init A: ||A||~1.65, cond~1.07):")
    print(f"{'client':14s} {'||A||':>6s} {'cond(A)':>8s} {'||A-A0||/||A0||':>16s} {'||B||':>7s} {'||dW||':>7s}")
    for name, (f, init) in clients.items():
        na = np.median([np.linalg.norm(f[c]["A"]) for c in cells]); ca = np.median([np.linalg.cond(f[c]["A"]) for c in cells])
        dr = np.median([np.linalg.norm(f[c]["A"] - init[c]) / np.linalg.norm(init[c]) for c in cells])
        nb = np.median([np.linalg.norm(f[c]["B"]) for c in cells]); nd = np.median([np.linalg.norm(f[c]["B"] @ f[c]["A"]) for c in cells])
        print(f"{name:14s} {na:6.2f} {ca:8.2f} {dr:16.3f} {nb:7.3f} {nd:7.3f}")
    for regime in ("S", "I", "I2"):
        names = [n for n in clients if n.endswith("_" + regime)]
        if len(names) < 2:
            print(f"\n[{regime}] fewer than 2 clients trained yet ({names})"); continue
        print(f"\n[{regime}] pairwise mean principal-angle cosine, by projection (chance ± sd from random subspaces):")
        for proj in ("q_proj", "k_proj", "v_proj"):
            pc = [c for c in cells if c[1] == proj]
            simA, simB, simDA = [], [], []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    fi, ii = clients[names[i]]; fj, ij = clients[names[j]]
                    for c in pc:
                        simA.append(mean_cos(orthobasis(fi[c]["A"].T), orthobasis(fj[c]["A"].T)))
                        simB.append(mean_cos(orthobasis(fi[c]["B"]), orthobasis(fj[c]["B"])))
                        simDA.append(mean_cos(orthobasis((fi[c]["A"] - ii[c]).T), orthobasis((fj[c]["A"] - ij[c]).T)))
            r = clients[names[0]][0][pc[0]]["A"].shape[0]
            cA, sA = chance_level(clients[names[0]][0][pc[0]]["A"].shape[1], r)
            cB, sB = chance_level(clients[names[0]][0][pc[0]]["B"].shape[0], r)
            print(f"  {proj}: A-rows {np.mean(simA):.3f} (chance {cA:.3f}±{sA:.3f}, +{(np.mean(simA)-cA)/sA:.1f}σ) | "
                  f"B-cols {np.mean(simB):.3f} (chance {cB:.3f}±{sB:.3f}, +{(np.mean(simB)-cB)/sB:.1f}σ) | "
                  f"A-DRIFT-rows {np.mean(simDA):.3f} (+{(np.mean(simDA)-cA)/sA:.1f}σ)")
    same = [(a, b) for a in clients for b in clients if a < b and a[:-2] == b[:-2]]
    if same:
        print("\nsame task, S vs I (same data & recipe, different A_0):")
        for a, b in same:
            fa, _ = clients[a]; fb, _ = clients[b]
            sa = np.mean([mean_cos(orthobasis(fa[c]["A"].T), orthobasis(fb[c]["A"].T)) for c in cells])
            sb = np.mean([mean_cos(orthobasis(fa[c]["B"]), orthobasis(fb[c]["B"])) for c in cells])
            print(f"  {a} vs {b}: A-rows {sa:.3f} | B-cols {sb:.3f}")

if __name__ == "__main__":
    main()
