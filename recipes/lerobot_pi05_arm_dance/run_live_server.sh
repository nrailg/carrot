#!/usr/bin/env bash
set -euo pipefail

ROOT="${MY_DFS:?confirm MY_DFS}/experiments/carrot/lerobot_pi05_arm_dance"
RUN="${ROOT}/${LIVE_RUN_NAME:?set a new run name}"
CHECKPOINT="${ROOT}/matched_20261009T124600Z/training/checkpoints/002000/pretrained_model"
SOURCE="${MY_DFS}/work/carrot/recipes/lerobot_pi05_arm_dance"
mkdir "$RUN"
mkdir "$RUN/source"
cp "$SOURCE/"*.py "$SOURCE/run_live_server.sh" "$RUN/source/"
source /opt/venvs/lerobot-official-arm-dance/bin/activate
unset PYTHONPATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
uv pip check --python /opt/venvs/lerobot-official-arm-dance/bin/python
bash /root/dguard/dguard.sh status
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
bash /root/dguard/dguard.sh stop 30
cleanup() {
    bash /root/dguard/dguard.sh on
}
trap cleanup EXIT
cd "$RUN/source"
uv pip freeze --python /opt/venvs/lerobot-official-arm-dance/bin/python > "$RUN/packages.txt"
date -u +'%FT%TZ' > "$RUN/started_at.txt"
python serve.py --checkpoint "$CHECKPOINT" --output "$RUN/server"
