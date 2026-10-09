# OCAP: polar capping for one-shot LoRA adapter merging

OCAP merges a pool of LoRA client adapters in one shot. Per layer and projection, each client's
update is first smoothed, then capped at a robust ceiling (median + k·MAD of the client norms), and the
capped pool is merged with a base rule (Iso-CTS by default). Code for building the merges, evaluating
them, and the baselines they are compared against.

The method lives in `src/merge.py` (`merge_adaptive`, `cap_ratio_pool`, `mad_tau`); the main
configuration is the rule `adaptive_mad` in `RULES`.

## Setup

```bash
# Python 3.13
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CUDA 12.6 torch wheels; edit the index URL for other CUDA versions
pytest tests                           # data-dependent tests skip until the data below is fetched
```

Models and adapters are downloaded from the Hugging Face Hub on first use:
`NousResearch/Meta-Llama-3.1-8B-Instruct` with the FinLoRA adapters `wangd12/*_llama_3_1_8b_8bits_r8`;
`mistralai/Mistral-7B-v0.1` with `predibase/*` adapters for LoRA Land (gated: run `hf auth login` and
accept the model's terms first); `llava-hf/llava-1.5-7b-hf` for MM-MergeBench.

The GSM8K eval for LoRA Land (`scripts/eval/loraland_gsm8k.sh`) uses lm-eval from a separate env:
`pip install lm_eval==0.4.13` there and set `LMEVAL_PYTHON` to its python.

## Data

```bash
scripts/data/fetch_external.sh         # FinLoRA, FinQA, ConvFinQA, ECTSum at pinned commits, plus XBRL CSVs
python scripts/data/ingest_heldout.py  # builds data/heldout/{finqa,convfinqa,ectsum,finexam10k}/test.jsonl
```

Two sources are fetched by hand:

- **FinExam10k** (arXiv 2608.28155): place `finexam10k_public_5110_canonical.jsonl` in `external/FinExam10k/`,
  or set `FINEXAM_SRC` to its path. Without it, ingestion skips finexam10k. The data is licensed for
  non-commercial research only.
- **MM-MergeBench** (only for the LLaVA experiments): the dataset
  [AuroraZengfh/MM-MergeBench](https://huggingface.co/datasets/AuroraZengfh/MM-MergeBench) goes in
  `external/mm_mergebench/data/`, the adapters `AuroraZengfh/LLaVA_7B_lora_r16_*` go in
  `external/mm_mergebench/adapters/`, and the images (COCO2014, ScienceQA, VizWiz, ImageNet, OCR-VQA,
  Flickr30k, IconQA, ImageNet-R, Screen2words) go in `external/mm_mergebench/images/`, as described in the
  [RobustMerge](https://github.com/AuroraZengfh/RobustMerge) README.

## Quick start

```bash
# merge the 8 FinLoRA r8 adapters with OCAP and two baselines
python scripts/merge/make_merges.py --rules adaptive_mad,isocts,ties
# evaluate the OCAP merge on the in-pool tasks and the held-out suite
python scripts/eval/run_eval.py --model merged:results/finlora/merges/main/adaptive_mad --tasks all --n 100 \
    --out results/finlora/pool/ocap
python scripts/eval/run_eval.py --model merged:results/finlora/merges/main/adaptive_mad --suite heldout \
    --tasks finqa,convfinqa,finexam10k --n 200 --out results/finlora/heldout/ocap
```

## Layout

```
src/                 merge rules (merge.py, asym.py), adapter pools (pool.py, lmm_pool.py),
                     tasks and scoring (tasks.py, heldout.py), eval runner (runner.py, stamps.py),
                     repo paths and the results/ layout (paths.py)
scripts/
  data/              external data fetch, held-out suite ingestion and audit
  merge/             build merged adapters (FinLoRA, LOO, rho/n-clients sweeps, MM, LoRA Land)
  eval/              evaluation entry points (run_eval.py, lmm_eval.py, loraland_eval.py, lm-eval wrappers)
  analysis/          aggregation, bootstrap tests, diagnostics, campaign verification
tests/               pytest suite
external/            third-party benchmarks (not tracked; see Data)
data/heldout/        built held-out test sets (not tracked)
results/             merged adapters, metrics and logs written by the scripts (not tracked)
```

Code addresses result directories only through `src.paths.res("<name>", ...)`, where `<name>` is the
short experiment name (`merges`, `lmm_eval`, ...); `src/paths.py` maps names to locations.
