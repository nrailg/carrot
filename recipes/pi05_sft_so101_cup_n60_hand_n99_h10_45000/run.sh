#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: bash run.sh --config /absolute/run/train.yaml [--check-only] [--full]" >&2
}

CONFIG=""
CHECK_ONLY=0
FULL=0
while (($#)); do
  case "$1" in
    --config)
      if (($# < 2)) || [[ -n "$CONFIG" ]]; then
        usage
        exit 2
      fi
      CONFIG="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --full)
      FULL=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
if [[ "$CONFIG" != /* || ! -f "$CONFIG" ]]; then
  echo "--config must name an existing absolute YAML path" >&2
  exit 2
fi

RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$RECIPE_DIR/../.." && pwd -P)
if [[ ! -f "$REPO_ROOT/pyproject.toml" || ! -f "$REPO_ROOT/src/carrot/cli/train_sft.py" ]]; then
  echo "run.sh requires a complete Carrot source checkout at $REPO_ROOT" >&2
  exit 2
fi
cd -- "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_EXECUTABLE="${PYTHON:-python}"
PREFLIGHT_MODULE="recipes.pi05_sft_so101_cup_n60_hand_n99_h10_45000.preflight"
PREFLIGHT_ARGS=(--config "$CONFIG")
if ((FULL)); then
  PREFLIGHT_ARGS+=(--full)
fi

if ((CHECK_ONLY)); then
  exec "$PYTHON_EXECUTABLE" -m "$PREFLIGHT_MODULE" "${PREFLIGHT_ARGS[@]}"
fi
case "${RAY_ADDRESS:-}" in
  ""|[Aa][Uu][Tt][Oo]|[Ll][Oo][Cc][Aa][Ll]|*[[:space:]]*)
    echo "Set RAY_ADDRESS to the explicit address of an externally managed Ray cluster" >&2
    exit 2
    ;;
esac

"$PYTHON_EXECUTABLE" -m "$PREFLIGHT_MODULE" "${PREFLIGHT_ARGS[@]}" --reserve-output
exec "$PYTHON_EXECUTABLE" -u -m carrot.cli.train_sft --config "$CONFIG"
