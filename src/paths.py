"""Repository paths and the layout of results/.

Result directories are addressed by their short name (the name each one had before
results/ was grouped by experiment family), e.g. res("e1_pool2000", "base__metrics.json").
Short names stay stable even if the on-disk layout moves again; only RESULT_DIRS changes.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

_FINLORA_POOL = ["e1_pool", "e1_pool500", "e1_pool2000", "e1_pool_seed1", "e1_pool500_seed7",
                 "e1_pool2000_seed7", "e8", "e8_500", "e8_2000", "clean_pool", "clean_pool500",
                 "chat_pool500", "loo_pool", "loo_pool500", "rho_pool", "rslora_pool", "nclients_pool",
                 "repro_bs1", "repro_bs8", "prompt_smoke"]
_FINLORA_HELDOUT = ["e1_heldout", "e1_heldout500", "e1_heldout2000", "e1_heldout2000_seed7",
                    "e8_heldout", "clean_heldout", "chat_heldout500", "rslora_heldout"]
_FINLORA_CAPABILITY = ["e6", "e6_500", "e6_2000", "e6_500_rslora"]

# short name -> path relative to results/
RESULT_DIRS = {
    # FinLoRA pool: merged adapters, then evaluations on the pool tasks, held-out suite and lm-eval capability
    "merges": "finlora/merges/main",
    "merges_loo": "finlora/merges/loo",
    "merges_nclients": "finlora/merges/nclients",
    "merges_rho": "finlora/merges/rho",
    "merges_rslora": "finlora/merges/rslora",
    **{d: f"finlora/pool/{d}" for d in _FINLORA_POOL},
    **{d: f"finlora/heldout/{d}" for d in _FINLORA_HELDOUT},
    **{d: f"finlora/capability/{d}" for d in _FINLORA_CAPABILITY},
    # N5: self-trained clients from a shared init
    "n5": "n5/clients",
    "n5_merges": "n5/merges",
    "n5_pool": "n5/pool",
    "n5_pool500": "n5/pool500",
    "n5_heldout": "n5/heldout",
    "n5_heldout500": "n5/heldout500",
    # LoRA Land (Predibase Mistral-7B pool)
    "loraland_merges": "loraland/merges",
    "loraland": "loraland/eval",
    "loraland_gsm8k": "loraland/gsm8k",
    # MM-MergeBench (LLaVA-1.5-7B)
    "lmm_merges": "mm/merges",
    "lmm_merges_rho": "mm/merges_rho",
    "lmm_eval": "mm/eval",
    "lmm_rho": "mm/eval_rho",
    # cross-pool diagnostics, figure caches, summary outputs, run logs
    "d1": "d1",
    "figdata": "summary/figdata",
    "no_regression.json": "summary/no_regression.json",
    "bootstrap_matrix.txt": "summary/bootstrap_matrix.txt",
    "logs": "logs",
}


def res_rel(name, *rest):
    """Path of a result directory relative to the repository root, e.g. 'results/finlora/pool/e1_pool'."""
    if name not in RESULT_DIRS:
        raise KeyError(f"unknown result directory {name!r}; add it to src/paths.py RESULT_DIRS")
    return "/".join(["results", RESULT_DIRS[name], *rest])


def res(name, *rest):
    """Absolute path of a result directory (or a file inside it)."""
    return os.path.join(ROOT, res_rel(name, *rest))
