#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
export CARROT_SO101_DATASET="${MY_DFS}/hf-hub/felixmayor/orange_cube_merged"
export CARROT_SO101_KNOCK_DOWN_DATASET="${MY_DFS}/hf-hub/nrailg/so101_knock_down_the_cylinder"
[[ -f "${CARROT_SO101_DATASET}/meta/info.json" ]]
[[ -f "${CARROT_SO101_KNOCK_DOWN_DATASET}/meta/info.json" ]]
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_dataset.py "$@"
