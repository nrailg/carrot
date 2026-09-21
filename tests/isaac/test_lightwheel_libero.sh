#!/usr/bin/env bash
set -euo pipefail

: "${MY_DFS:?Set MY_DFS to the verified Gemini personal DFS directory}"
CARROT_DIR="${MY_DFS}/work/carrot"
LW_EXAMPLE_VENV="${LW_EXAMPLE_VENV:-/opt/venvs/lightwheel-libero}"
LW_EXAMPLE_OUTPUT="${LW_EXAMPLE_OUTPUT:-${MY_DFS}/benchmarks/lightwheel-libero-20260920/smoke}"
source "${LW_EXAMPLE_VENV}/bin/activate"
export PYTHONPATH="${CARROT_DIR}/src:${CARROT_DIR}/tests:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${CARROT_DIR}"
python examples/isaac/04_lightwheel_libero.py \
    --headless --enable_cameras --num_envs 1 --steps 60 --output "$LW_EXAMPLE_OUTPUT"
python tests/isaac/test_lightwheel_libero.py "$LW_EXAMPLE_OUTPUT" --steps 60
