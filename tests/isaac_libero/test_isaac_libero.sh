#!/usr/bin/env bash
set -euo pipefail

: "${MY_DFS:?Set MY_DFS from the current Gemini session}"
: "${ISAAC_LIBERO_OUTPUT:?Set a new persistent output directory under MY_DFS}"
CARROT_DIR="${MY_DFS}/work/carrot"
source "${ISAAC_LIBERO_VENV:-/opt/venvs/lightwheel-libero}/bin/activate"
export PYTHONPATH="${CARROT_DIR}/src:${CARROT_DIR}/tests:${PYTHONPATH:-}"
if [[ -n "${ISAAC_LIBERO_ARENA_ROOT:-}" ]]; then
    export PYTHONPATH="${ISAAC_LIBERO_ARENA_ROOT}:${PYTHONPATH}"
fi
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$CARROT_DIR"
if [[ -e "$ISAAC_LIBERO_OUTPUT/result.json" ]]; then
    echo 'Use a new output directory; refusing to reuse a previous completion marker.' >&2
    exit 1
fi
python -m pytest -q tests/isaac_libero/test_isaac_libero.py
python tests/isaac_libero/run_isaac_libero.py \
    --headless --enable_cameras --num_envs 4 --steps 32 \
    --kit_args='--/renderer/multiGpu/enabled=false' \
    --output "$ISAAC_LIBERO_OUTPUT"
python tests/isaac_libero/test_isaac_libero.py --result "$ISAAC_LIBERO_OUTPUT"
