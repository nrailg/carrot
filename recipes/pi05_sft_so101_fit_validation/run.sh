#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm MY_DFS on the current GPU server}"
: "${RAY_ADDRESS:?verify Ray before running the fit suite}"
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash /root/dguard/dguard.sh stop 360
trap 'bash /root/dguard/dguard.sh on' EXIT
python -u recipes/pi05_sft_so101_fit_validation/suite.py "$@"
