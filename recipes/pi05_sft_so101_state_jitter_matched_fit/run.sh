#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm MY_DFS on Gemini}"
: "${RAY_ADDRESS:?verify Ray before launch}"
: "${SOURCE_COMMIT:?pass the verified Mac source commit}"
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CASE_DIR="$PWD/recipes/pi05_sft_so101_state_jitter_matched_fit"
OUTPUT_ROOT=$(python -c 'import pathlib,sys,yaml; print(pathlib.Path(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["output_dir"]).parent)' "$CASE_DIR/train.yaml")
test ! -e "$OUTPUT_ROOT/training"
test ! -e "$OUTPUT_ROOT/evaluation"
bash /root/dguard/dguard.sh stop 120
trap 'bash /root/dguard/dguard.sh on' EXIT
python -u recipes/pi05_sft_so101_fit_validation/train.py --config "$CASE_DIR/train.yaml"
export CUDA_VISIBLE_DEVICES=0
for STEP in 500 1000 1500 2000; do
    python -u recipes/pi05_sft_so101_state_only_rollout/evaluate.py --case-dir "$CASE_DIR" --step "$STEP" --output "$OUTPUT_ROOT/evaluation/step$STEP" --source-commit "$SOURCE_COMMIT"
    python -u recipes/pi05_sft_so101_state_only_rollout/verify.py --output "$OUTPUT_ROOT/evaluation/step$STEP" --source-data "$MY_DFS/hf-hub/nrailg/knock_down_the_cylinder_1_20260930_222251/data"
done
