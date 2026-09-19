#!/usr/bin/env bash
set -Eeuo pipefail
: "${CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT:?set the downloaded PI0.5 checkpoint directory}"
[[ -f "${CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT}/config.json" ]]
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_pi05_checkpoint_distributed.py "$@"
