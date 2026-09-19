#!/usr/bin/env bash
set -Eeuo pipefail
bash "$(dirname "$0")/../run_case.sh" tests/sft/test_sft_checkpoint_distributed.py "$@"
