#!/usr/bin/env bash
set -Eeuo pipefail

: "${MY_DFS:?set MY_DFS from the current Gemini session}"
: "${SO101_USD:?set SO101_USD to the local official PhysX SO101 asset}"
export SO101_USD
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
python -m pytest -v -s tests/isaac_so101/test_motion.py
