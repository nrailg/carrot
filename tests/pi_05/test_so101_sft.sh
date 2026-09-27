#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_so101_sft.py "$@"
