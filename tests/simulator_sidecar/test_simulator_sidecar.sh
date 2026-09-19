#!/usr/bin/env bash
set -Eeuo pipefail
bash "$(dirname "$0")/../run_case.sh" tests/simulator_sidecar/test_simulator_sidecar.py "$@"
