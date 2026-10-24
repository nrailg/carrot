#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm current DFS}"
source /opt/venvs/carrot/bin/activate
cd "${CARROT_SOURCE_DIR:-$MY_DFS/work/carrot}"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
python -m pytest tests/libero_draw_transfer/test_schedule.py -q
