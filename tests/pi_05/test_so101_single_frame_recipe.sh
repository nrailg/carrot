#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?set MY_DFS to the current personal DFS root}"
export PYTHONPATH="${MY_DFS}/work/carrot:${PYTHONPATH:-}"
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_single_frame_recipe.py "$@"
