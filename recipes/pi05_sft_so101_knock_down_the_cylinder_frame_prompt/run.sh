#!/usr/bin/env bash
set -Eeuo pipefail
: "${MY_DFS:?confirm MY_DFS on Gemini}"
: "${RAY_ADDRESS:?verify Ray before launch}"
: "${SOURCE_COMMIT:?pass the verified Mac source commit}"
[[ "$MY_DFS" == /mnt/ceph-hz1-csp/mm-base-plt2/nrwu ]] || {
    echo "update the pinned train.yaml paths for the current MY_DFS before launch" >&2
    exit 1
}
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CASE_DIR="$PWD/recipes/pi05_sft_so101_knock_down_the_cylinder_frame_prompt"
OUTPUT_ROOT=$(python -c 'import pathlib,sys,yaml; print(pathlib.Path(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["output_dir"]).parent)' "$CASE_DIR/train.yaml")
CALIBRATION=$(python -c 'import pathlib,sys,yaml; print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["dataset"]["factory_kwargs"]["state_jitter_calibration_path"])' "$CASE_DIR/train.yaml")
test -s "$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch_h10/model.safetensors"
test -s "$MY_DFS/hf-hub/google/paligemma-3b-pt-224/tokenizer.json"
test -f "$MY_DFS/hf-hub/nrailg/knock_down_the_cylinder_1_20260930_222251/meta/info.json"
test -f "$CALIBRATION"
test ! -e "$OUTPUT_ROOT"
test ! -L "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
cp "$CASE_DIR/train.yaml" "$OUTPUT_ROOT/train.yaml"
cp "$CALIBRATION" "$OUTPUT_ROOT/calibration.json"
printf '%s\n' "$SOURCE_COMMIT" > "$OUTPUT_ROOT/source_commit.txt"
bash /root/dguard/dguard.sh stop 120
trap 'bash /root/dguard/dguard.sh on' EXIT
python -u recipes/pi05_sft_so101_fit_validation/train.py --config "$CASE_DIR/train.yaml"
