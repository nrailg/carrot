#!/usr/bin/env bash
set -Eeuo pipefail

CARROT_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/carrot"
OPENPI_DIR="${MY_DFS}/work/openpi"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${MY_DFS}/hf-hub/Physical-Intelligence/pi05_libero_pytorch}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${MY_DFS}/hf-hub/google/paligemma-3b-pt-224}"
DATASET_ROOT="${DATASET_ROOT:-${MY_DFS}/hf-hub/lerobot/libero}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
POLICY_GPU="${POLICY_GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MY_DFS}/benchmarks/carrot-pi05-libero-parity}"
RUN_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d-%H%M%S)"

[[ -f "${CARROT_DIR}/tests/pi_05/test_libero_parity.py" ]]
[[ -f "${CHECKPOINT_DIR}/model.safetensors" ]]
[[ -f "${CHECKPOINT_DIR}/assets/physical-intelligence/libero/norm_stats.json" ]]
[[ -d "$TOKENIZER_DIR" ]]
[[ -f "$TOKENIZER_DIR/tokenizer.model" ]]
[[ -d "$DATASET_ROOT" ]]
[[ -x /opt/venvs/openpi-libero/bin/python ]]
[[ -x /opt/venvs/carrot/bin/python ]]
[[ -f "${OPENPI_DIR}/examples/libero/main.py" ]]

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "$RUN_DIR"
GOLDEN="$RUN_DIR/openpi-libero-golden.npz"
OBSERVATION="$RUN_DIR/libero-observation.npz"

(
    source /opt/venvs/carrot/bin/activate
    cd "$CARROT_DIR"
    export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
    test "$(command -v python)" = /opt/venvs/carrot/bin/python
    python tests/pi_05/capture_libero_observation.py \
        --dataset-root "$DATASET_ROOT" \
        --repo-id "$DATASET_REPO_ID" \
        --sample-index "$SAMPLE_INDEX" \
        --output "$OBSERVATION"
)

(
    source /opt/venvs/openpi-libero/bin/activate
    cd "$OPENPI_DIR"
    export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU"
    test "$(command -v python)" = /opt/venvs/openpi-libero/bin/python
    python "$CARROT_DIR/tests/pi_05/generate_openpi_libero_golden.py" \
        --openpi-dir "$OPENPI_DIR" \
        --checkpoint "$CHECKPOINT_DIR" \
        --tokenizer-model "$TOKENIZER_DIR/tokenizer.model" \
        --observation "$OBSERVATION" \
        --num-steps 10 \
        --output "$GOLDEN"
)

(
    source /opt/venvs/carrot/bin/activate
    cd "$CARROT_DIR"
    export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$POLICY_GPU"
    export CARROT_PI05_LIBERO_GOLDEN="$GOLDEN"
    export CARROT_PI05_LIBERO_CHECKPOINT="$CHECKPOINT_DIR"
    export CARROT_PI05_LIBERO_TOKENIZER="$TOKENIZER_DIR"
    test "$(command -v python)" = /opt/venvs/carrot/bin/python
    python -m pytest -v -s --timeout=1800 tests/pi_05/test_libero_parity.py
)

printf 'Gate 2 artifact: %s\n' "$GOLDEN"
