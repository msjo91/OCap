#!/bin/bash
set -o pipefail; cd "$(dirname "$0")/../.."
GPU=$1; shift
OUT=results/loraland/gsm8k
for m in "$@"; do
  extra=""; [ "$m" != base ] && extra=",peft=results/loraland/merges/$m"
  [ -f $OUT/$m/done ] && continue
  CUDA_VISIBLE_DEVICES=$GPU ${LMEVAL_PYTHON:-python} -m lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-v0.1,dtype=bfloat16,load_in_8bit=True$extra \
    --tasks gsm8k --limit 500 --batch_size 8 --output_path $OUT/$m --log_samples 2>&1 | grep -v -E "(User|Future|Deprecation)Warning" | tail -20
  if [ "${PIPESTATUS[0]}" = 0 ]; then touch $OUT/$m/done; else echo "GSM8K FAILED $m"; fi
done
echo "LORALAND GSM8K GPU$GPU done $(date -u +%H:%M)"
