#!/usr/bin/env bash
set -Eeuo pipefail
bash "$(dirname "$0")/../run_case.sh" tests/distributed_runtime/test_model_parallelizer.py "$@"
