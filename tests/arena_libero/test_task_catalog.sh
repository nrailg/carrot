#!/usr/bin/env bash
set -Eeuo pipefail
CARROT_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/carrot"
source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMNI_KIT_ACCEPT_EULA=YES
python -m pytest -q tests/arena_libero/test_arena_libero.py tests/arena_libero/test_task_catalog.py
test ! -e "${ARENA_LIBERO_OUTPUT:?}"
mkdir -p "$ARENA_LIBERO_OUTPUT"
mapfile -t TASK_IDS < <(python -c 'from carrot_sim.arena_libero.tasks import list_tasks; print("\n".join(t.task_id for t in list_tasks()))')
for task_id in "${TASK_IDS[@]}"; do
  python tests/arena_libero/run_task_catalog.py \
    --asset-root "${ARENA_LIBERO_ASSETS:?}" \
    --output "${ARENA_LIBERO_OUTPUT:?}/${task_id}" --task-id "$task_id" \
    --headless --enable_cameras --kit_args='--/renderer/multiGpu/enabled=false'
  test -f "${ARENA_LIBERO_OUTPUT}/${task_id}/result.json"
done
python tests/arena_libero/test_task_catalog.py "$ARENA_LIBERO_OUTPUT"
