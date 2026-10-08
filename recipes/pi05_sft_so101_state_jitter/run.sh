#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm MY_DFS on Gemini}"
: "${RAY_ADDRESS:?verify Ray before launch}"
: "${SOURCE_COMMIT:?pass the verified Mac source commit}"
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CASE_DIR="$PWD/recipes/pi05_sft_so101_state_jitter"
OUTPUT_ROOT=$(python -c 'import pathlib,sys,yaml; print(pathlib.Path(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["output_dir"]).parent)' "$CASE_DIR/train.yaml")
STEP=$(python -c 'import pathlib,sys,yaml; print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["steps"])' "$CASE_DIR/train.yaml")
test ! -e "$OUTPUT_ROOT/training"
test ! -e "$OUTPUT_ROOT/evaluation"
mkdir -p "$OUTPUT_ROOT"
cp "$CASE_DIR/train.yaml" "$OUTPUT_ROOT/train.yaml"
CALIBRATION=$(python -c 'import pathlib,sys,yaml; print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["dataset"]["factory_kwargs"]["state_jitter_calibration_path"])' "$CASE_DIR/train.yaml")
cp "$CALIBRATION" "$OUTPUT_ROOT/calibration.json"
bash /root/dguard/dguard.sh stop 120
trap 'bash /root/dguard/dguard.sh on' EXIT
python -u recipes/pi05_sft_so101_fit_validation/train.py --config "$CASE_DIR/train.yaml"
export CUDA_VISIBLE_DEVICES=0
python -u recipes/pi05_sft_so101_state_only_rollout/evaluate.py --case-dir "$CASE_DIR" --step "$STEP" --output "$OUTPUT_ROOT/evaluation" --source-commit "$SOURCE_COMMIT"
python -u recipes/pi05_sft_so101_state_only_rollout/verify.py --output "$OUTPUT_ROOT/evaluation" --source-data "$MY_DFS/hf-hub/nrailg/knock_down_the_cylinder_1_20260930_222251/data"
