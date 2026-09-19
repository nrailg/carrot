#!/usr/bin/env bash
set -Eeuo pipefail

CARROT_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/carrot"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${MY_DFS}/hf-hub/Physical-Intelligence/pi05_libero_pytorch}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${MY_DFS}/hf-hub/google/paligemma-3b-pt-224}"
DATASET_ROOT="${DATASET_ROOT:-${MY_DFS}/hf-hub/lerobot/libero}"
POLICY_GPU="${POLICY_GPU:-0}"

[[ -f "$CARROT_DIR/tests/pi_05/test_libero_sft_alignment.py" ]]
[[ -f "$CHECKPOINT_DIR/model.safetensors" ]]
[[ -f "$CHECKPOINT_DIR/assets/physical-intelligence/libero/norm_stats.json" ]]
[[ -f "$TOKENIZER_DIR/tokenizer.model" ]]
[[ -f "$DATASET_ROOT/meta/info.json" ]]

source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$POLICY_GPU"
export CARROT_PI05_LIBERO_CHECKPOINT="$CHECKPOINT_DIR"
export CARROT_PI05_LIBERO_TOKENIZER="$TOKENIZER_DIR"
export CARROT_PI05_LIBERO_DATASET_ROOT="$DATASET_ROOT"
test "$(command -v python)" = /opt/venvs/carrot/bin/python
python -m pytest -v -s --timeout=1800 tests/pi_05/test_libero_sft_alignment.py
