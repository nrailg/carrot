#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
: "${CARROT_SO101_CHECKPOINT:?set a SO101 checkpoint}"
export CARROT_SO101_CHECKPOINT
export CARROT_SO101_DATASET="${MY_DFS}/hf-hub/nrailg/so101_knock_down_the_cylinder"
export CARROT_SO101_WEBSOCKET_RUN="${MY_DFS}/test-runs/so101-websocket-$(date -u +%Y%m%dT%H%M%S)-$$"
[[ -r "${CARROT_SO101_CHECKPOINT}/model.safetensors" ]]
[[ -r "${CARROT_SO101_CHECKPOINT}/norm_stats.json" ]]
[[ -r "${CARROT_SO101_CHECKPOINT}/tokenizer_config.json" ]]
[[ -r "${CARROT_SO101_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
printf 'Artifacts: %s\n' "$CARROT_SO101_WEBSOCKET_RUN"
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_websocket.py "$@"
