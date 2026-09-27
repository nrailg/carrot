#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
export CARROT_SO101_CHECKPOINT="${MY_DFS}/experiments/carrot/pi05_sft_so101_orange_cube/checkpoints/step-00005000"
export CARROT_SO101_DATASET="${MY_DFS}/hf-hub/felixmayor/orange_cube_merged"
export CARROT_SO101_WEBSOCKET_RUN="${MY_DFS}/test-runs/so101-websocket-$(date -u +%Y%m%dT%H%M%S)-$$"
[[ -r "${CARROT_SO101_CHECKPOINT}/model.safetensors" ]]
[[ -r "${CARROT_SO101_CHECKPOINT}/norm_stats.json" ]]
[[ -r "${CARROT_SO101_CHECKPOINT}/tokenizer_config.json" ]]
[[ -r "${CARROT_SO101_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
printf 'Artifacts: %s\n' "$CARROT_SO101_WEBSOCKET_RUN"
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_websocket.py "$@"
