#!/usr/bin/env bash
set -Eeuo pipefail
bash "$(dirname "$0")/../run_case.sh" tests/data_loading/test_data_loading.py "$@"
