#!/usr/bin/env bash
set -Eeuo pipefail
CARROT_DIR="${MY_DFS:?confirm learner DFS}/work/carrot"
: "${POLICY_CONFIG:?resolved experiment YAML on the learner node}"
: "${TEST_OUTPUT:?fresh persistent output directory}"
: "${POLICY_GPU:?explicit idle compute GPU}"
source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$POLICY_GPU"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python tests/multicolor_transfer/test_policy.py --config "$POLICY_CONFIG" --output "$TEST_OUTPUT"
