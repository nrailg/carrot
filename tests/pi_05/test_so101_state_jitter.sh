#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS:?confirm MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q tests/pi_05/test_so101_state_jitter.py
