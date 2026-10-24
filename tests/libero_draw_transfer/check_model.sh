#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm current DFS}"
: "${TRAIN_CONFIG:?prepared mixed configuration}"
: "${TEST_OUTPUT:?new persistent output}"
source /opt/venvs/carrot/bin/activate
cd "${CARROT_SOURCE_DIR:-$MY_DFS/work/carrot}"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OMP_NUM_THREADS=4 torchrun --standalone --nproc-per-node=8 \
  tests/libero_draw_transfer/check_model.py --config "$TRAIN_CONFIG" --output "$TEST_OUTPUT"
