#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm learner DFS}"
: "${TEST_OUTPUT:?fresh persistent output}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
if [[ "${1:-}" == --credit-input ]]; then
  exec python tests/multicolor_transfer/test_reward_signal.py --output "$TEST_OUTPUT" "$@"
fi
: "${POLICY_CONFIG:?resolved config}"
: "${POLICY_CHECKPOINT:?trusted SFT checkpoint}"
: "${POLICY_GPU:?explicit idle compute GPU}"
export CUDA_VISIBLE_DEVICES="$POLICY_GPU"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python tests/multicolor_transfer/test_reward_signal.py --config "$POLICY_CONFIG" \
  --checkpoint "$POLICY_CHECKPOINT" --output "$TEST_OUTPUT" "$@"
