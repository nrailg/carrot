#!/usr/bin/env bash
set -euo pipefail
: "${MY_DFS:?Detect MY_DFS from the current Gemini session}"
: "${ARENA_LIBERO_ASSETS:?Set the local USD asset directory}"
: "${ARENA_LIBERO_OUTPUT:?Set a new persistent output directory}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m pytest -q tests/arena_libero/test_arena_libero.py
python tests/arena_libero/run_arena_libero.py \
    --asset-root "$ARENA_LIBERO_ASSETS" --output "$ARENA_LIBERO_OUTPUT" \
    --headless --enable_cameras --kit_args='--/renderer/multiGpu/enabled=false'
python tests/arena_libero/test_arena_libero.py "$ARENA_LIBERO_OUTPUT"
