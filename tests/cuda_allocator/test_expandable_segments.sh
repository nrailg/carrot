#!/usr/bin/env bash
set -Eeuo pipefail

CARROT_DIR="${MY_DFS:?set MY_DFS to the current personal DFS root}/work/carrot"
RUN_DIR="${RUN_DIR:-${MY_DFS}/benchmarks/carrot-expandable-segments/$(date +%Y%m%d-%H%M%S)}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-30}"
TEST_FILE="tests/cuda_allocator/test_expandable_segments.py"

[[ -f "${CARROT_DIR}/${TEST_FILE}" ]]
[[ -x /opt/venvs/carrot/bin/python ]]
mkdir -p "$RUN_DIR"

source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
test "$(command -v python)" = /opt/venvs/carrot/bin/python

unset PYTORCH_ALLOC_CONF
for mode in false true; do
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:${mode^}" \
        python "$TEST_FILE" benchmark \
        --expected-expandable "$mode" \
        --device "$CUDA_DEVICE" \
        --warmup "$WARMUP" \
        --iterations "$ITERATIONS" \
        --output "$RUN_DIR/$mode.json" \
        >"$RUN_DIR/$mode.log" 2>&1
done

python "$TEST_FILE" compare \
    --baseline "$RUN_DIR/false.json" \
    --expandable "$RUN_DIR/true.json" \
    --output "$RUN_DIR/summary.json" \
    | tee "$RUN_DIR/summary.log"

printf 'RESULT_DIR=%s\n' "$RUN_DIR"
