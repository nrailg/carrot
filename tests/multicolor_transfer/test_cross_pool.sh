#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirmed H20 DFS}"
: "${TEST_OUTPUT:?fresh learner test output}"
: "${SIMULATOR_TEST_OUTPUT:?fresh renderer test output}"
: "${RAY_ADDRESS:?start both explicitly placed Ray nodes first}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/examples/isaac_workbench:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python tests/multicolor_transfer/test_cross_pool.py \
  --config recipes/pi05_multicolor_transfer/train.yaml \
  --output "$TEST_OUTPUT" --simulator-output "$SIMULATOR_TEST_OUTPUT"
