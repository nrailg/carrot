#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm current DFS}"
: "${RAY_ADDRESS:?verify an eight-GPU Ray cluster}"
: "${RUN_DIR:?new persistent experiment directory}"
: "${DRAW_ROOT:?verified new demonstrations}"
: "${TEACHER_GATE:?16-episode physical preflight JSON}"
carrot_dir="${CARROT_SOURCE_DIR:-$MY_DFS/work/carrot}"
source /opt/venvs/carrot/bin/activate
cd "$carrot_dir"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python recipes/pi05_libero_draw_sft_transfer/prepare.py \
  --output "$RUN_DIR" --draw-root "$DRAW_ROOT" --teacher-gate "$TEACHER_GATE"
TRAIN_CONFIG="$RUN_DIR/mixed.yaml" TEST_OUTPUT="$RUN_DIR/gpu-preflight" \
  bash tests/libero_draw_transfer/check_model.sh
python -m carrot.cli.train_sft --config "$RUN_DIR/libero.yaml"
python -m carrot.cli.train_sft --config "$RUN_DIR/mixed.yaml"
