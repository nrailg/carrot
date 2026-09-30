#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
: "${SO101_DIAG_OUTPUT:?set a new output directory}"
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/diagnose.py \
  --checkpoint-joint-units degrees \
  --checkpoint "$MY_DFS/experiments/carrot/pi05_sft_so101_wipe_overfit_100/checkpoints/step-00000100" \
  --dataset-root "$MY_DFS/hf-hub/nrailg/so101_knock_down_the_cylinder" \
  --tokenizer "$MY_DFS/hf-hub/google/paligemma-3b-pt-224" \
  --previous-evaluation "$MY_DFS/experiments/carrot/pi05_sft_so101_wipe_overfit_100/evaluation/step100" \
  --output "$SO101_DIAG_OUTPUT"
