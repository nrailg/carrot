#!/usr/bin/env bash
set -Eeuo pipefail
: "${CARROT_PI05_MODEL_PATH:?set the downloaded PI0.5 model directory}"
: "${CARROT_PI05_TOKENIZER_PATH:?set the downloaded tokenizer directory}"
: "${CARROT_ROBOTWIN_ROOT:?set the downloaded RoboTwin dataset directory}"
[[ -f "${CARROT_PI05_MODEL_PATH}/config.json" ]]
[[ -f "${CARROT_PI05_TOKENIZER_PATH}/tokenizer.model" ]]
[[ -d "$CARROT_ROBOTWIN_ROOT" ]]
bash "$(dirname "$0")/../run_case.sh" tests/sft/test_sft_trainer.py "$@"
