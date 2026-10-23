#!/usr/bin/env bash
set -Eeuo pipefail
CARROT_DIR="${MY_DFS:?confirm renderer DFS}/work/carrot"
: "${TEST_OUTPUT:?fresh persistent output directory}"
: "${SIMULATOR_GPU:?explicit idle Pro5000 GPU}"
ROBOT_USD="${MY_DFS}/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
[[ -f "$ROBOT_USD" ]]
nvidia-smi -i "$SIMULATOR_GPU" --query-gpu=name --format=csv,noheader | grep -q 'RTX PRO 5000'
source /opt/venvs/carrot/bin/activate
cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/examples/isaac_workbench:${PYTHONPATH:-}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
[[ -f "$VK_ICD_FILENAMES" ]]
python examples/isaac_workbench/run.py --task multicolor --headless --gpu "$SIMULATOR_GPU" \
  --robot-usd "$ROBOT_USD" --output "${TEST_OUTPUT}-hold" --policy hold --steps 50 \
  --num-envs 2 --num-colors 2 --partial-reset-check --check
python examples/isaac_workbench/run.py --task multicolor --headless --gpu "$SIMULATOR_GPU" \
  --robot-usd "$ROBOT_USD" --output "$TEST_OUTPUT" --policy scripted --steps 4000 \
  --num-envs 2 --num-colors 2 --partial-reset-check --check
python tests/multicolor_transfer/test_physics.py --output "$TEST_OUTPUT"
