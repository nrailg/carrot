#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS from the current Gemini session}"
: "${CARROT_SO101_CHECKPOINT:?set a SO101 checkpoint with degree/percentage statistics}"
export CARROT_SO101_CHECKPOINT
export CARROT_SO101_DATASET="${CARROT_SO101_DATASET:-${MY_DFS}/hf-hub/nrailg/so101_knock_down_the_cylinder}"
[[ -f "${CARROT_SO101_CHECKPOINT}/model.safetensors" ]]
[[ -f "${CARROT_SO101_CHECKPOINT}/norm_stats.json" ]]
[[ -f "${CARROT_SO101_CHECKPOINT}/tokenizer_config.json" ]]
[[ -f "${CARROT_SO101_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_inference_checkpoint.py "$@"
