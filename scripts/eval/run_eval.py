import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src import tasks as T
from src import heldout as H
from src.pool import adapter_repo
from src.runner import eval_model_on_tasks, eval_model_on_heldout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--suite", default="pool", choices=["pool", "heldout"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--prompt_style", default="raw", choices=["raw", "chat"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.model == "base":
        name, adapter = "base", None
    elif args.model.startswith("client:"):
        parts = args.model.split(":")
        task, pool = parts[1], (parts[2] if len(parts) > 2 else "r8")
        name, adapter = (f"client_{task}" if pool == "r8" else f"client_{task}_{pool}"), adapter_repo(task, pool)
    elif args.model.startswith("merged:"):
        path = args.model.split(":", 1)[1]
        name, adapter = "merged_" + os.path.basename(path.rstrip("/")), path
    elif args.model.startswith("local:"):
        path = args.model.split(":", 1)[1]
        name, adapter = os.path.basename(path.rstrip("/")), path
    elif args.model.startswith("hf:"):
        repo = args.model.split(":", 1)[1]
        name, adapter = repo.replace("/", "-"), repo
    else:
        raise SystemExit(f"bad --model {args.model}")

    if args.suite == "heldout":
        task_names = list(H.TASKS) if args.tasks == "all" else args.tasks.split(",")
        eval_model_on_heldout(name, adapter, task_names, args.n, args.out,
                              seed=args.seed, force=args.force, clean=args.clean,
                              prompt_style=args.prompt_style)
    else:
        task_names = list(T.TASKS) if args.tasks == "all" else args.tasks.split(",")
        eval_model_on_tasks(name, adapter, task_names, args.n, args.out,
                            seed=args.seed, batch_size=args.batch_size,
                            force=args.force, clean=args.clean,
                            prompt_style=args.prompt_style)

if __name__ == "__main__":
    main()
