#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm MY_DFS on Gemini}"
: "${RAY_ADDRESS:?verify Ray before launch}"
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash /root/dguard/dguard.sh stop 60
trap 'bash /root/dguard/dguard.sh on' EXIT
python -u recipes/pi05_sft_so101_fit_validation/train.py --config recipes/pi05_sft_so101_state_jitter/train.yaml
export CUDA_VISIBLE_DEVICES=0
python -u recipes/pi05_sft_so101_state_only_rollout/evaluate.py --case-dir recipes/pi05_sft_so101_state_jitter --output "$MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261004T033515Z/evaluation" --source-commit 71b1144
python -u recipes/pi05_sft_so101_state_only_rollout/verify.py --output "$MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261004T033515Z/evaluation" --source-data "$MY_DFS/hf-hub/nrailg/knock_down_the_cylinder_1_20260930_222251/data"
