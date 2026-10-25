#!/usr/bin/env bash
set -Eeuo pipefail
source /opt/venvs/carrot/bin/activate
cd "${CARROT_SOURCE_DIR:-${MY_DFS:?}/work/carrot}"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
python -m pytest -q tests/libero_draw_transfer/test_eval_pairing.py
