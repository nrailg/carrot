#!/usr/bin/env bash
set -Eeuo pipefail
: "${CARROT_SO101_DATASET:?set the downloaded orange_cube_merged directory}"
[[ -f "${CARROT_SO101_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_dataset.py "$@"
