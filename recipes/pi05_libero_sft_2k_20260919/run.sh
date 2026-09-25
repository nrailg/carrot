#!/usr/bin/env bash
set -Eeuo pipefail

# This historical config pins paths under this DFS root. Copy and edit the config
# before starting a new experiment on another root.
expected_dfs=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
my_dfs="${MY_DFS:?resolve MY_DFS in the current Gemini session}"
[[ "$my_dfs" == "$expected_dfs" ]] || {
    echo "the pinned training config uses $expected_dfs; copy and update it for $my_dfs" >&2
    exit 1
}
: "${RAY_ADDRESS:?start and verify the Ray cluster before running this recipe}"

carrot_dir="$my_dfs/work/carrot"
run_dir="$my_dfs/experiments/carrot/pi05_libero_sft_2k_20260919"
config="$carrot_dir/configs/pi05_sft_libero_2k_20260919.yaml"

[[ -f "$config" ]]
[[ -s "$run_dir/init/model.safetensors" ]]
[[ -s "$my_dfs/hf-hub/google/paligemma-3b-pt-224/tokenizer.json" ]]
[[ -f "$my_dfs/hf-hub/lerobot/libero/meta/info.json" ]]
[[ -s "$my_dfs/hf-hub/Physical-Intelligence/pi05_libero_pytorch/assets/physical-intelligence/libero/norm_stats.json" ]]
[[ ! -e "$run_dir/checkpoints" ]] || {
    echo "existing checkpoints in $run_dir; use a new output_dir for a new run" >&2
    exit 1
}

source /opt/venvs/carrot/bin/activate
cd "$carrot_dir"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ "$(command -v python)" == /opt/venvs/carrot/bin/python ]]

python -m carrot.cli.train_sft --config "$config"
