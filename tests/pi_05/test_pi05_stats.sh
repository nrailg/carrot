#!/usr/bin/env bash
set -Eeuo pipefail
bash "$(dirname "$0")/../run_case.sh" tests/pi_05/test_pi05_stats.py "$@"
