#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm current learner DFS}"
: "${LEARNER_DFS:?set confirmed H20 DFS}"
: "${SIMULATOR_DFS:?set confirmed Pro5000 DFS}"
: "${LEARNER_NODE_IP:?set H20 Ray node IP}"
: "${SIMULATOR_NODE_IP:?set Pro5000 Ray node IP}"
: "${RUN_NAME:?choose a fresh persistent run name}"
: "${RAY_ADDRESS:?start the explicit two-node cluster first}"
[[ "$MY_DFS" == "$LEARNER_DFS" ]]
source /opt/venvs/carrot/bin/activate
cd "${CARROT_SOURCE_DIR:-${LEARNER_DFS}/work/carrot}"
export PYTHONPATH="$PWD/src:$PWD/examples/isaac_workbench:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -m carrot.cli.train_rl --config recipes/pi05_multicolor_transfer/train.yaml "$@"
