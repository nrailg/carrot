#!/usr/bin/env bash
set -euo pipefail

CARROT_DIR=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
DATASET_ROOT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/lerobot/libero

source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

python examples/lerobot/inspect_dataset.py \
  --repo-id lerobot/libero \
  --root "$DATASET_ROOT" \
  --index 0
