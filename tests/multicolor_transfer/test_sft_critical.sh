#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm learner DFS}"
: "${POLICY_CONFIG:?resolved experiment config}"
: "${POLICY_CHECKPOINT:?trusted periodic SFT checkpoint}"
: "${TEST_OUTPUT:?fresh persistent output directory}"
: "${POLICY_GPU:?explicit idle GPU}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$POLICY_GPU"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python tests/multicolor_transfer/test_sft_critical.py --config "$POLICY_CONFIG" \
  --checkpoint "$POLICY_CHECKPOINT" --output "$TEST_OUTPUT" "$@"
