#!/usr/bin/env bash
set -euo pipefail
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS:?confirm MY_DFS on Gemini}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u recipes/pi05_sft_so101_fit_validation/reevaluate.py "$@"
