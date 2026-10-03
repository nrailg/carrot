#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm MY_DFS on the current GPU server}"
export PYTHONPATH="${MY_DFS}/work/carrot:${PYTHONPATH:-}"
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_fit_controls.py "$@"
