#!/usr/bin/env bash
# Clone the external benchmarks into external/ at the commits used for the reported results,
# then fetch the XBRL training CSVs that FinLoRA keeps on the Hugging Face Hub.
# FinExam10k and MM-MergeBench are not fetched here; see README "Data".
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p external

fetch() {  # name url commit
    local dir="external/$1"
    if [ -d "$dir/.git" ]; then echo "$1: present, skipping"; return; fi
    git init -q "$dir"
    git -C "$dir" remote add origin "$2"
    git -C "$dir" fetch -q --depth 1 origin "$3"
    git -C "$dir" checkout -q FETCH_HEAD
    echo "$1: $(git -C "$dir" rev-parse --short HEAD)"
}

fetch FinLoRA   https://github.com/Open-Finance-Lab/FinLoRA.git 6f9f32e137f18bec7148bfbdb593328d4c29b2cc
fetch FinQA     https://github.com/czyssrs/FinQA.git            0f16e2867befa6840783e58be38c9efb9229d742
fetch ConvFinQA https://github.com/czyssrs/ConvFinQA.git        cf3eed2d5984960bf06bb8145bcea5e80b0222a6
fetch ECTSum    https://github.com/rajdeep345/ECTSum.git        6909f1fc543104c1c60cf9de63e799f6620d1b0a

# ConvFinQA ships its data as a zip
[ -f external/ConvFinQA/data/dev.json ] || unzip -q -o external/ConvFinQA/data.zip -d external/ConvFinQA

# xbrl_extract training sizes (merge_concat_weighted) and the data audit read these CSVs
${PYTHON:-python} - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ["formula_calculation_train.csv", "formula_formatted_with_tags_train.csv",
          "value_train.csv", "xbrl_tags_train.csv"]:
    hf_hub_download("wangd12/XBRL_analysis", f, repo_type="dataset",
                    revision="6ee79f739a6c51d4ec39921693ee3d7e0cd54c17",
                    local_dir="external/FinLoRA/data/train/xbrl_csv")
print("XBRL_analysis: ok")
EOF
