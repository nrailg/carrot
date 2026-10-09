#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/opt/venvs/lerobot-official-arm-dance
RUN_DIR="${MY_DFS:?confirm MY_DFS}/experiments/carrot/lerobot_pi05_arm_dance/${RUN_ID:?set unique RUN_ID}"
export SOURCE_COMMIT="${SOURCE_COMMIT:?set verified Mac commit}"
export HF_HOME="$RUN_DIR/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export INITIAL_MODEL_DIR="$RUN_DIR/init_model"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
unset PYTHONPATH
source "$VENV/bin/activate"
cd "$RECIPE_DIR"
python preflight.py --run-dir "$RUN_DIR"
uv pip check --python "$VENV/bin/python"
cd "$RUN_DIR/source"

bash /root/dguard/dguard.sh stop "${DGUARD_MINUTES:-360}"
restore_guard() {
    local result=$?
    printf '%s\n' "$result" > "$RUN_DIR/exit_code.txt"
    bash /root/dguard/dguard.sh on
    exit "$result"
}
trap restore_guard EXIT
python evaluate.py --base --limit 3 --output "$RUN_DIR/base_play"

TRAIN_ARGS=(
    --dataset.repo_id=nrailg/arm_dance_20261008_225311_20261008_225313
    "--dataset.root=$MY_DFS/hf-hub/nrailg/arm_dance_20261008_225311_20261008_225313"
    --dataset.video_backend=pyav
    --policy.type=pi05_control
    --policy.empty_cameras=2
    --policy.optimizer_lr=1e-6
    --policy.optimizer_weight_decay=1e-10
    --policy.optimizer_betas="[0.9,0.95]"
    --policy.optimizer_eps=1e-8
    --policy.optimizer_grad_clip_norm=1.0
    --policy.scheduler_decay_lr=1e-6
    "--policy.pretrained_path=$INITIAL_MODEL_DIR"
    --policy.chunk_size=10
    --policy.n_action_steps=1
    --policy.dtype=bfloat16
    --policy.device=cuda
    --policy.gradient_checkpointing=true
    --policy.push_to_hub=false
    --policy.scheduler_warmup_steps=100
    "--policy.scheduler_decay_steps=${STEPS:-2000}"
    --batch_size=8
    --num_workers=2
    "--steps=${STEPS:-2000}"
    --save_freq=500
    --log_freq=1
    --env_eval_freq=0
    --wandb.enable=false
    --seed=1000
    "--output_dir=$RUN_DIR/training"
    --job_name=lerobot_pi05_arm_dance
)
printf '%q ' python -m torch.distributed.run --standalone --nproc_per_node=8 \
    train.py "${TRAIN_ARGS[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    train.py "${TRAIN_ARGS[@]}"
printf '0\n' > "$RUN_DIR/train_exit_code.txt"
python evaluate.py --checkpoint "$RUN_DIR/training/checkpoints/last/pretrained_model" \
    --output "$RUN_DIR/offline_play"
printf 'OFFICIAL TRAIN AND OFFLINE PLAY FINISHED\n'
