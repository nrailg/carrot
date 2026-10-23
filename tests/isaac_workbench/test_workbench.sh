#!/usr/bin/env bash
set -euo pipefail
: "${MY_DFS:?Set the verified personal DFS directory}"
: "${WORKBENCH_GUEST:?Set the prepared Linux guest directory}"
: "${WORKBENCH_OUTPUT:?Set a fresh persistent output directory}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export WORKBENCH_GUEST OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/isaac_workbench/test_workbench.py
ROBOT_USD="${MY_DFS}/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
for task in computer paint; do
    num_envs=2
    if [[ "$task" == computer ]]; then num_envs=3; fi
    python examples/isaac_workbench/run.py --task "$task" --headless \
        --gpu "${WORKBENCH_GPU:-1}" --robot-usd "$ROBOT_USD" \
        --guest-dir "$WORKBENCH_GUEST" --accelerator "${WORKBENCH_ACCELERATOR:-kvm}" \
        --num-envs "$num_envs" --exercise sum_two,max_list,reverse_text \
        --output "${WORKBENCH_OUTPUT}/${task}" --check --partial-reset-check
done
python examples/isaac_workbench/run.py --task paint --headless \
    --gpu "${WORKBENCH_GPU:-1}" --robot-usd "$ROBOT_USD" \
    --num-envs 2 --policy hold --steps 30 --partial-reset-check \
    --output "${WORKBENCH_OUTPUT}/hold" --check
