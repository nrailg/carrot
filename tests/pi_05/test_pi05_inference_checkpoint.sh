#!/usr/bin/env bash
set -Eeuo pipefail
: "${CARROT_PI05_INFERENCE_CHECKPOINT:?set the downloaded RoboTwin SFT export directory}"
[[ -f "${CARROT_PI05_INFERENCE_CHECKPOINT}/config.json" ]]
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_pi05_inference_checkpoint.py "$@"
