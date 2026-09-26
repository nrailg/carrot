#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS from the current Gemini session}"
export CARROT_SO101_CHECKPOINT="${CARROT_SO101_CHECKPOINT:-${MY_DFS}/experiments/carrot/pi05_sft_so101_orange_cube/checkpoints/step-00005000}"
export CARROT_SO101_DATASET="${CARROT_SO101_DATASET:-${MY_DFS}/hf-hub/felixmayor/orange_cube_merged}"
[[ -f "${CARROT_SO101_CHECKPOINT}/model.safetensors" ]]
[[ -f "${CARROT_SO101_CHECKPOINT}/norm_stats.json" ]]
[[ -f "${CARROT_SO101_CHECKPOINT}/tokenizer_config.json" ]]
[[ -f "${CARROT_SO101_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_inference_checkpoint.py "$@"
