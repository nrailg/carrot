#!/usr/bin/env bash
set -euo pipefail

GIGA_MODELS_COMMIT=b0e05c7353a2d5d06fa8a92d2103c0fdd52e2e4d
GIGA_MODELS_REPO=${GIGA_MODELS_REPO:-/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/giga-models-pi05-parity}
OPENPI_CHECKPOINT=${OPENPI_CHECKPOINT:-/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base}
TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Miical/pi05-base}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/checkpoints/open-gigaai/pi05_base-b0e05c7-fp32}

actual_commit=$(git -C "$GIGA_MODELS_REPO" rev-parse HEAD)
if [[ "$actual_commit" != "$GIGA_MODELS_COMMIT" ]]; then
  echo "Expected giga-models $GIGA_MODELS_COMMIT, got $actual_commit" >&2
  exit 1
fi

# TOKENIZER_PATH 只提供 save_pretrained 所需文件，不参与 JAX 模型参数转换。
python "$GIGA_MODELS_REPO/projects/vla/pi0/scripts/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir "$OPENPI_CHECKPOINT/params" \
  --precision float32 \
  --tokenizer-id "$TOKENIZER_PATH" \
  --output-path "$OUTPUT_PATH" \
  --pi05-enabled
