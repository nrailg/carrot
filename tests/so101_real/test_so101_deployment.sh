#!/usr/bin/env bash
set -Eeuo pipefail
bash "$(dirname "$0")/../run_case.sh" tests/so101_real/test_so101_deployment.py "$@"
