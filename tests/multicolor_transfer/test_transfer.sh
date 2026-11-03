#!/usr/bin/env bash
set -Eeuo pipefail
CARROT_DIR="${MY_DFS:?set MY_DFS from this Gemini session}/work/carrot"
source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/examples/isaac_workbench:${PYTHONPATH:-}"
python -m pytest -v tests/multicolor_transfer/test_transfer.py tests/distributed_runtime/test_placement.py
