#!/usr/bin/env bash
set -Eeuo pipefail

expected_dfs=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
my_dfs="${MY_DFS:?resolve MY_DFS in the current Gemini session}"
[[ "$my_dfs" == "$expected_dfs" ]] || {
    echo "the pinned training config uses $expected_dfs; copy and update it for $my_dfs" >&2
    exit 1
}
: "${RAY_ADDRESS:?start and verify the Ray cluster before running this recipe}"

carrot_dir="$my_dfs/work/carrot"
recipe_dir="$carrot_dir/recipes/pi05_sft_so101_knock_down_the_cylinder_overfit"
dataset_dir="$my_dfs/hf-hub/nrailg/knock_down_the_cylinder_1_20260930_222251"
model_dir="$my_dfs/hf-hub/Physical-Intelligence/pi05_base_pytorch_h10"
run_dir="$my_dfs/experiments/carrot/pi05_sft_so101_knock_down_the_cylinder_overfit_h10_5000"

[[ -f "$recipe_dir/train.yaml" ]]
[[ -s "$model_dir/model.safetensors" ]]
[[ -f "$model_dir/config.json" ]]
[[ -s "$my_dfs/hf-hub/google/paligemma-3b-pt-224/tokenizer.json" ]]
[[ -f "$dataset_dir/meta/info.json" ]] || {
    echo "missing local SO101 dataset: $dataset_dir" >&2
    exit 1
}
[[ ! -e "$run_dir" && ! -L "$run_dir" ]] || {
    echo "output already exists: $run_dir; use a new output_dir or explicit resume" >&2
    exit 1
}

source /opt/venvs/carrot/bin/activate
cd "$carrot_dir"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[[ "$(command -v python)" == /opt/venvs/carrot/bin/python ]]

python -m carrot.cli.train_sft --config "$recipe_dir/train.yaml"
