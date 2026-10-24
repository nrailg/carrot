#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm current DFS}"
source /opt/venvs/carrot/bin/activate
cd "${CARROT_SOURCE_DIR:-$MY_DFS/work/carrot}"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD/examples/isaac_workbench:${PYTHONPATH:-}"
python -m pytest tests/libero_draw_transfer/test_static_task.py tests/libero_draw_transfer/test_data.py -q
