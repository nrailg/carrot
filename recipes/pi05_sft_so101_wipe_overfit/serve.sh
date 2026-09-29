#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
carrot_dir="$MY_DFS/work/carrot"
checkpoint="$MY_DFS/experiments/carrot/pi05_sft_so101_wipe_overfit_100/checkpoints/step-00000100"
[[ -s "$checkpoint/model.safetensors" ]]
[[ -s "$checkpoint/norm_stats.json" ]]
source /opt/venvs/carrot/bin/activate
cd "$carrot_dir"
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u -m carrot.cli.serve_pi05_policy \
  --embodiment so101 --checkpoint "$checkpoint" --device cuda:0 \
  --host 0.0.0.0 --port "${PORT:-8080}" --default-prompt 'Move an object'
