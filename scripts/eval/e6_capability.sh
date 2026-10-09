#!/bin/bash
set -o pipefail; cd "$(dirname "$0")/../.."
run(){
  local extra=""; [ -n "$2" ] && extra=",peft=$2"
  if [ -n "$2" ] && [ -d "$2" ] && [ ! -f "$2/adapter_model.safetensors" ]; then echo "E6 FAILED $1: adapter dir without weights"; FAILED=1; return; fi
  if [ -f $OUT/$1/done ]; then
    if [ -z "$2" ] || [ ! -d "$2" ] || [ $OUT/$1/done -nt "$2/adapter_model.safetensors" ]; then echo "skip $1"; return; fi
    echo "redo $1 (adapter newer than marker)"
  fi
  CUDA_VISIBLE_DEVICES=$GPU ${LMEVAL_PYTHON:-python} -m lm_eval --model hf --model_args pretrained=NousResearch/Meta-Llama-3.1-8B-Instruct,dtype=bfloat16,load_in_8bit=True$extra \
    --tasks mmlu,gsm8k,hellaswag --limit $LIMIT --batch_size 8 --output_path $OUT/$1 --log_samples 2>&1 | grep -v -E "(User|Future|Deprecation)Warning" | tail -60
  if [ "${PIPESTATUS[0]}" = 0 ]; then touch $OUT/$1/done; else echo "E6 FAILED $1"; FAILED=1; fi
}
GPU=$1; FAILED=0
LIMIT=${E6_LIMIT:-200}; OUT=${E6_OUT:-results/finlora/capability/e6}
if [ -n "$E6_MODELS" ]; then
  for spec in $E6_MODELS; do run "${spec%%=*}" "${spec#*=}"; done
else
  run base ""
  run client_xbrl_term wangd12/xbrl_term_llama_3_1_8b_8bits_r8
  run concat results/finlora/merges/main/concat
  run ties64_g50 results/finlora/merges/main/ties64_g50
  run normeq_med results/finlora/merges/main/normeq_med
fi
if [ "$FAILED" = 1 ]; then echo "E6 GPU $GPU done WITH FAILURES"; exit 1; fi
echo "E6 GPU $GPU done"
