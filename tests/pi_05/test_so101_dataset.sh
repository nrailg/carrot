#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
export CARROT_SO101_DATASET="${MY_DFS}/hf-hub/felixmayor/orange_cube_merged"
[[ -f "${CARROT_SO101_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_dataset.py "$@"
