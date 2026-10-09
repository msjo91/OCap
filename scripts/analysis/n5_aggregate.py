import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.paths import res
from src import tasks as T, heldout as H
from src.stamps import GEN_CFG, heldout_gen_cfg, generation_is_current
from scripts.merge.n5_merges import pool_members, DEFAULT_RULES, is_self_trained, member_task

RNG = np.random.default_rng(0)
LABEL_TASKS = [t for t, (f, _, m, own) in T.TASKS.items() if m.startswith("label_acc")]
LABELS = {t: T.load_task(t, n=None)[1] for t in LABEL_TASKS}
HELD = ["finqa", "convfinqa", "finexam10k"]
STALE = []

def score_rows(task, rows):
    outs = [r["output"] for r in rows]; tg = [r["target"] for r in rows]
    if T.TASKS[task][2] == "label_acc_batched":
        outs, tg = T.split_batched(outs, tg)
    return [int(T.score_task(task, [o], [t], LABELS[task])["acc"]) for o, t in zip(outs, tg)]

def adapter_dir(model):
    
    if model.startswith("merged_"):
        return res("n5_merges", model[len("merged_"):])
    if is_self_trained(model):
        return res("n5", model)
    return None

def weights_current(rec, model):
    
    d = adapter_dir(model)
    if d is None:
        return True
    f = os.path.join(d, "adapter_model.safetensors")
    if not os.path.exists(f):
        return False
    stamp = round(os.path.getmtime(f), 1)
    return rec.get("adapter_mtime", stamp) == stamp

def collect(d, res):
    for f in glob.glob(f"{d}/*__*.jsonl"):
        m, t = os.path.basename(f)[:-6].split("__", 1)
        if t not in LABEL_TASKS:
            continue
        mf = os.path.join(d, f"{m}__metrics.json")
        rec = json.load(open(mf)).get(t) if os.path.exists(mf) else None
        if not generation_is_current(rec, t, GEN_CFG, "pool") or not weights_current(rec, m):
            STALE.append((m, t)); continue
        res.setdefault(m, {})[t] = score_rows(t, [json.loads(l) for l in open(f)])
    return res

def collect_heldout(d, res):
    for f in glob.glob(f"{d}/*__metrics.json"):
        for t, v in json.load(open(f)).items():
            if t not in HELD:
                continue
            if not generation_is_current(v, t, heldout_gen_cfg(H.TASKS[t], t), "heldout") or not weights_current(v, v["model"]):
                STALE.append((v["model"], t)); continue
            res.setdefault(v["model"], {})[t] = v.get("acc")
    return res

def member_task_owner(member):
    return member_task(member)

def member_model(member):
    return member if is_self_trained(member) else "client_" + member

acc = lambda c: float(np.mean(c)) if c else float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pools", default="S,I,mixed,S_light,I_light,S_fz,S_lr01"); args = ap.parse_args()
    pool_gens = {}; collect(res("e8"), pool_gens); collect(res("n5_pool"), pool_gens)
    held = {}; collect_heldout(res("e8_heldout"), held); collect_heldout(res("n5_heldout"), held)
    if STALE:
        print(f"NOTE: {len(STALE)} stale (model, task) records skipped, e.g. {STALE[:3]}\n")
    base = pool_gens.get("base", {})
    for pname in args.pools.split(","):
        members = pool_members(pname)
        owners = sorted({member_task_owner(m) for m in members})
        tasks = [t for t in LABEL_TASKS if T.TASKS[t][3] in owners]
        rules = [r for r in DEFAULT_RULES if f"merged_{pname}-{r}" in pool_gens]
        if not rules:
            print(f"=== pool {pname}: no merged evals yet\n"); continue
        best, who = {}, {}
        for t in tasks:
            cands = [(acc(pool_gens.get(member_model(m), {}).get(t, [])), m) for m in members
                     if pool_gens.get(member_model(m), {}).get(t)]
            best[t], who[t] = (max(cands) if cands else (float("nan"), "-"))
        common = [t for t in tasks if all(t in pool_gens[f"merged_{pname}-{r}"] for r in rules) and t in base]
        print(f"=== pool {pname} ({len(members)} clients: {', '.join(members)}) — pool tasks n=100 (xbrl_finer n=400); "
              f"mean/ret/>=base over the {len(common)} tasks all rules share")
        hdr = ["model"] + [t[:11] for t in tasks] + ["mean", "ret", ">=base"]
        print(" ".join(f"{h:>11s}" for h in hdr))
        def row(name, vals):
            v = [vals.get(t, float("nan")) for t in tasks]
            cm = [vals[t] for t in common if t in vals]
            ret = np.nanmean([vals[t] / best[t] for t in common if t in vals and best[t] > 0]) if cm else float("nan")
            ge = sum(vals.get(t, -1) >= base_acc.get(t, 2) for t in common)
            print(f"{name[:11]:>11s} " + " ".join(f"{x:11.3f}" for x in v)
                  + f" {(np.mean(cm) if len(cm) == len(common) and cm else float('nan')):11.3f} {ret:11.2f} {ge:11d}")
        base_acc = {t: acc(base.get(t, [])) for t in tasks}
        row("base", base_acc)
        for m in members:
            g = pool_gens.get(member_model(m), {})
            if any(t in g for t in tasks):
                row(m, {t: acc(g[t]) for t in tasks if t in g})
        row("best-member", best)
        print(" " * 12 + " ".join(f"{who[t][:11]:>11s}" for t in tasks))
        rule_acc = {}
        for r in rules:
            g = pool_gens[f"merged_{pname}-{r}"]
            rule_acc[r] = {t: acc(g[t]) for t in tasks if t in g}
            row(r, rule_acc[r])
        def paired(a, b):
            per = {}
            for t in tasks:
                ga, gb = pool_gens[f"merged_{pname}-{a}"].get(t), pool_gens[f"merged_{pname}-{b}"].get(t)
                if ga and gb and len(ga) == len(gb):
                    per[t] = np.array(ga) - np.array(gb)
            if not per:
                return None
            d = np.concatenate(list(per.values()))
            bs = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            tb = [np.mean([per[t][RNG.integers(0, len(per[t]), len(per[t]))].mean() for t in per]) for _ in range(2000)]
            return (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d),
                    np.mean([per[t].mean() for t in per]), np.percentile(tb, 2.5), np.percentile(tb, 97.5))
        pairs = [(r, "concat") for r in rules if r != "concat"]
        if "regime_switch" in rules:
            pairs += [("regime_switch", r) for r in ("trim_g50", "avg_factors") if r in rules]
        print("paired bootstrap (delta acc, 95% CI): pooled examples | task-balanced (equal task weights)")
        for a, b in pairs:
            p = paired(a, b)
            if p:
                print(f"  {a:14s} - {b:12s} {p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}] n={p[3]} | {p[4]:+.3f} [{p[5]:+.3f}, {p[6]:+.3f}]")
        print("held-out (n=200):")
        print(f"{'model':>14s}" + "".join(f"{t:>12s}" for t in HELD) + f"{'mean':>8s}")
        def hrow(name, vals):
            v = [vals.get(t) for t in HELD]
            got = [x for x in v if x is not None]
            print(f"{name[:14]:>14s}" + "".join(f"{x:12.3f}" if x is not None else f"{'-':>12s}" for x in v)
                  + (f"{np.mean(got):8.3f}" if got else f"{'-':>8s}"))
        if "base" in held:
            hrow("base", held["base"])
        for m in members:
            if member_model(m) in held:
                hrow(m, held[member_model(m)])
        for r in rules:
            k = f"merged_{pname}-{r}"
            if k in held:
                hrow(r, held[k])
        print()

if __name__ == "__main__":
    main()
