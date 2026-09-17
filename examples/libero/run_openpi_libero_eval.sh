#!/usr/bin/env bash
# see: docs/test-manifest/openpi-libero-official-benchmark-20260916.md

set -Eeuo pipefail

TASK_SUITE="libero_spatial"
TASK_ID=0
EPISODES=100
SERVER_GPU=0
RENDER_GPU=2
PORT=8000
SEED=7
REPLAN_STEPS=5
SERVER_TIMEOUT_SECONDS=900
DGUARD_STOP_MINUTES=120

CHECKPOINT_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_libero_pytorch"
# CHECKPOINT_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base_pytorch_libero_eval"

OPENPI_DATA_HOME="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub"
TOKENIZER_PATH="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/big_vision/paligemma_tokenizer.model"
DEFAULT_PROXY_URL="http://star-proxy.oa.com:3128"
TOKENIZER_URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_DOWNLOAD_TIMEOUT="${OPENPI_TOKENIZER_DOWNLOAD_TIMEOUT:-300}"

usage() {
    cat <<'EOF'
Run one OpenPI PyTorch policy and one LIBERO evaluator with isolated GPUs.

Usage:
  run_openpi_libero_eval.sh [options]

Options:
  --task-suite NAME       LIBERO suite (default: libero_spatial)
  --task-id ID            Run exactly one task (default: 0)
  --episodes N            Episodes for the selected task (default: 100)
  --server-gpu ID         Physical GPU for the policy server (default: 0)
  --render-gpu ID         Physical GPU for the EGL renderer (default: 2)
  --port PORT             Policy server port (default: 8000)
  --openpi-dir PATH       OpenPI checkout (default: $MY_DFS/work/openpi)
  --output-root PATH      Persistent result root
  --help                   Show this help

Environment:
  MY_DFS                   Personal CephFS root. Auto-detected only when there is
                           exactly one CephFS team directory.
  OPENPI_PROXY_URL         Outbound proxy (default: http://star-proxy.oa.com:3128)
  OPENPI_TOKENIZER_DOWNLOAD_TIMEOUT
                           Tokenizer download timeout in seconds (default: 300)
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

OPENPI_DIR=""
OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-suite)
            TASK_SUITE="${2:?missing value for --task-suite}"
            shift 2
            ;;
        --task-id)
            TASK_ID="${2:?missing value for --task-id}"
            shift 2
            ;;
        --episodes)
            EPISODES="${2:?missing value for --episodes}"
            shift 2
            ;;
        --server-gpu)
            SERVER_GPU="${2:?missing value for --server-gpu}"
            shift 2
            ;;
        --render-gpu)
            RENDER_GPU="${2:?missing value for --render-gpu}"
            shift 2
            ;;
        --port)
            PORT="${2:?missing value for --port}"
            shift 2
            ;;
        --openpi-dir)
            OPENPI_DIR="${2:?missing value for --openpi-dir}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?missing value for --output-root}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "$TASK_ID" =~ ^[0-9]+$ ]] || die "--task-id must be a non-negative integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "--episodes must be a positive integer"
[[ "$SERVER_GPU" =~ ^[0-9]+$ ]] || die "--server-gpu must be a non-negative integer"
[[ "$RENDER_GPU" =~ ^[0-9]+$ ]] || die "--render-gpu must be a non-negative integer"
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer"
[[ "$TOKENIZER_DOWNLOAD_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || \
    die "OPENPI_TOKENIZER_DOWNLOAD_TIMEOUT must be a positive integer"
[[ "$SERVER_GPU" != "$RENDER_GPU" ]] || die "policy server and EGL renderer must use different GPUs"

resolve_my_dfs() {
    if [[ -n "${MY_DFS:-}" ]]; then
        [[ -d "$MY_DFS" ]] || die "MY_DFS does not exist: $MY_DFS"
        return
    fi

    [[ -n "${__SYS_USER_NAME__:-}" ]] || die "MY_DFS is unset and __SYS_USER_NAME__ is unavailable"
    local candidates=()
    local path
    for path in /mnt/ceph-*-csp/*; do
        [[ -d "$path" ]] || continue
        [[ "$(findmnt -rn -T "$path" -o FSTYPE 2>/dev/null || true)" == "fuse.ceph-fuse" ]] || continue
        candidates+=("$path")
    done
    [[ ${#candidates[@]} -eq 1 ]] || die \
        "cannot safely auto-detect MY_DFS; set it explicitly (CephFS team candidates: ${candidates[*]:-none})"
    MY_DFS="${candidates[0]}/${__SYS_USER_NAME__}"
    [[ -d "$MY_DFS" ]] || die "detected MY_DFS does not exist: $MY_DFS"
}

resolve_my_dfs
OPENPI_DIR="${OPENPI_DIR:-${MY_DFS}/work/openpi}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MY_DFS}/benchmarks/openpi-libero}"

proxy_url="${OPENPI_PROXY_URL:-$DEFAULT_PROXY_URL}"
export http_proxy="${http_proxy:-$proxy_url}"
export https_proxy="${https_proxy:-$proxy_url}"
export ftp_proxy="${ftp_proxy:-$proxy_url}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export FTP_PROXY="${FTP_PROXY:-$ftp_proxy}"
proxy_bypass="localhost,127.0.0.1,.oa.com,.woa.com,mirrors.tencent.com"
export no_proxy="${no_proxy:+${no_proxy},}${proxy_bypass}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${proxy_bypass}"

VENV="/opt/venvs/openpi-libero"
DGUARD="/root/dguard/dguard.sh"
EVALUATOR="${OPENPI_DIR}/examples/libero/main.py"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${TASK_SUITE}-task${TASK_ID}-${EPISODES}x-${RUN_ID}"
SERVER_LOG="${RUN_DIR}/logs/server.log"
EVAL_LOG="${RUN_DIR}/logs/eval.log"
THERMAL_LOG="${RUN_DIR}/logs/thermal.csv"
RESULT_PATH="${RUN_DIR}/results/episodes.jsonl"
VIDEO_DIR="${RUN_DIR}/videos"

[[ -x "${VENV}/bin/python" ]] || die "missing Python environment: $VENV"
[[ -f "$DGUARD" ]] || die "missing dguard entrypoint: $DGUARD"
[[ -f "$EVALUATOR" ]] || die "missing LIBERO evaluator: $EVALUATOR"
[[ -f "${OPENPI_DIR}/scripts/serve_policy.py" ]] || die "missing OpenPI policy server"
[[ -s "${CHECKPOINT_DIR}/model.safetensors" ]] || die "missing PyTorch model weights"
[[ "$(stat -Lc %s "${CHECKPOINT_DIR}/model.safetensors")" -gt 7000000000 ]] || \
    die "PyTorch model weights appear incomplete"
[[ -f "${CHECKPOINT_DIR}/config.json" ]] || die "missing PyTorch model config"
[[ -f "${CHECKPOINT_DIR}/assets/physical-intelligence/libero/norm_stats.json" ]] || \
    die "missing LIBERO normalization stats"
if [[ ! -s "$TOKENIZER_PATH" ]]; then
    command -v curl >/dev/null || die "curl is required to prefetch the tokenizer"
    mkdir -p "$(dirname "$TOKENIZER_PATH")"
    tokenizer_partial="${TOKENIZER_PATH}.partial.$$"
    echo "Tokenizer cache is missing; downloading before any GPU work"
    if ! curl --fail --location --progress-bar \
        --connect-timeout 20 --max-time "$TOKENIZER_DOWNLOAD_TIMEOUT" \
        "$TOKENIZER_URL" --output "$tokenizer_partial"; then
        rm -f "$tokenizer_partial"
        die "tokenizer download failed or timed out before GPU startup"
    fi
    if ! "${VENV}/bin/python" -c \
        'import sentencepiece as spm, sys; spm.SentencePieceProcessor(model_file=sys.argv[1])' \
        "$tokenizer_partial"; then
        rm -f "$tokenizer_partial"
        die "downloaded tokenizer is invalid"
    fi
    mv "$tokenizer_partial" "$TOKENIZER_PATH"
fi
"${VENV}/bin/python" -c \
    'import sentencepiece as spm, sys; spm.SentencePieceProcessor(model_file=sys.argv[1])' \
    "$TOKENIZER_PATH" || die "tokenizer cache is unreadable: $TOKENIZER_PATH"
grep -q "task_id" "$EVALUATOR" || die "evaluator does not support --args.task-id"
grep -q "result_out_path" "$EVALUATOR" || die "evaluator does not support durable JSONL results"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
[[ "$SERVER_GPU" -lt "$(nvidia-smi --list-gpus | wc -l)" ]] || die "server GPU is out of range"
[[ "$RENDER_GPU" -lt "$(nvidia-smi --list-gpus | wc -l)" ]] || die "renderer GPU is out of range"
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
    die "port $PORT is already in use"
fi
[[ ! -e "$RUN_DIR" ]] || die "run directory already exists: $RUN_DIR"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/results" "$VIDEO_DIR"

source "${VENV}/bin/activate"
cd "$OPENPI_DIR"
export OPENPI_DATA_HOME
export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src:/opt/libero/src:${PYTHONPATH:-}"
export OPENPI_POLICY_DIR="$CHECKPOINT_DIR"

SERVER_PID=""
MONITOR_PID=""
DGUARD_PAUSED=0
XID_COUNT_BEFORE=""

cleanup() {
    local rc=$?
    local xid_count_after=""
    trap - EXIT INT TERM
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ "$DGUARD_PAUSED" -eq 1 ]]; then
        if [[ -n "$XID_COUNT_BEFORE" ]]; then
            xid_count_after="$(dmesg | grep -ci 'NVRM: Xid' || true)"
        fi
        if [[ -n "$xid_count_after" && "$xid_count_after" -gt "$XID_COUNT_BEFORE" ]]; then
            dmesg -T | grep -i 'NVRM: Xid' | tail -n "$((xid_count_after - XID_COUNT_BEFORE))" \
                | tee "${RUN_DIR}/logs/new-xid.log" || true
            bash "$DGUARD" off || true
            echo "New NVIDIA Xid detected; dguard remains off" >&2
            rc=1
        else
            bash "$DGUARD" on || true
            if [[ -n "$xid_count_after" ]]; then
                echo "No new NVIDIA Xid detected"
            else
                echo "NVIDIA Xid check unavailable"
            fi
        fi
    fi
    echo "run directory: $RUN_DIR"
    echo "server log:   $SERVER_LOG"
    echo "eval log:     $EVAL_LOG"
    echo "thermal log:  $THERMAL_LOG"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bash "$DGUARD" stop "$DGUARD_STOP_MINUTES"
DGUARD_PAUSED=1

gpu_pids() {
    nvidia-smi -i "$1" -q -d PIDS 2>/dev/null \
        | awk -F: '/Process ID/{gsub(/ /, "", $2); print $2}' \
        | sort -u
}

for _ in $(seq 1 15); do
    server_pids="$(gpu_pids "$SERVER_GPU")"
    render_pids="$(gpu_pids "$RENDER_GPU")"
    [[ -z "$server_pids" && -z "$render_pids" ]] && break
    sleep 1
done
[[ -z "$server_pids" ]] || die "server GPU $SERVER_GPU is already occupied by PID(s): $server_pids"
[[ -z "$render_pids" ]] || die "renderer GPU $RENDER_GPU is already occupied by PID(s): $render_pids"

if dmesg >/dev/null 2>&1; then
    XID_COUNT_BEFORE="$(dmesg | grep -ci 'NVRM: Xid' || true)"
fi

monitor_gpus() {
    echo "timestamp,gpu,pci_bus_id,temperature_gpu,temperature_memory,power_draw,memory_used,utilization_gpu"
    while true; do
        local timestamp
        timestamp="$(date --iso-8601=seconds)"
        nvidia-smi -i "${SERVER_GPU},${RENDER_GPU}" \
            --query-gpu=index,pci.bus_id,temperature.gpu,temperature.memory,power.draw,memory.used,utilization.gpu \
            --format=csv,noheader,nounits | sed "s/^/${timestamp},/"
        sleep 10
    done
}
monitor_gpus > >(tee "$THERMAL_LOG") &
MONITOR_PID=$!

echo "Starting PyTorch policy server on physical GPU $SERVER_GPU"
export OPENPI_SERVER_PORT="$PORT"
CUDA_VISIBLE_DEVICES="$SERVER_GPU" python -u - <<'PY' > >(tee "$SERVER_LOG") 2>&1 &
import os
import runpy
import sys

import torch

sys.argv = [
    "scripts/serve_policy.py",
    "--port=" + os.environ["OPENPI_SERVER_PORT"],
    "policy:checkpoint",
    "--policy.config=pi05_libero",
    "--policy.dir=" + os.environ["OPENPI_POLICY_DIR"],
]
print("RUNTIME_TORCH_COMPILE=eager_identity", flush=True)
torch.compile = lambda function, *args, **kwargs: function
runpy.run_path("scripts/serve_policy.py", run_name="__main__")
PY
SERVER_PID=$!

ready=0
for second in $(seq 1 "$SERVER_TIMEOUT_SECONDS"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        die "policy server exited before opening port $PORT"
    fi
    if python -c \
        'import os; from websockets.sync.client import connect; connection = connect("ws://127.0.0.1:" + os.environ["OPENPI_SERVER_PORT"], open_timeout=1, close_timeout=1); connection.close()' \
        2>/dev/null; then
        ready=1
        break
    fi
    if (( second % 30 == 0 )); then
        echo "Waiting for policy server: ${second}s/${SERVER_TIMEOUT_SECONDS}s"
    fi
    sleep 1
done
[[ "$ready" -eq 1 ]] || die "policy server did not open port $PORT within ${SERVER_TIMEOUT_SECONDS}s"

echo "Running ${TASK_SUITE} task ${TASK_ID}: ${EPISODES} episodes on physical GPU $RENDER_GPU"
CUDA_VISIBLE_DEVICES="$RENDER_GPU" \
MUJOCO_EGL_DEVICE_ID="$RENDER_GPU" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python -u "$EVALUATOR" \
    --args.host=127.0.0.1 \
    --args.port="$PORT" \
    --args.task-suite-name="$TASK_SUITE" \
    --args.task-id="$TASK_ID" \
    --args.num-trials-per-task="$EPISODES" \
    --args.replan-steps="$REPLAN_STEPS" \
    --args.seed="$SEED" \
    --args.video-out-path="$VIDEO_DIR" \
    --args.result-out-path="$RESULT_PATH" \
    2>&1 | tee "$EVAL_LOG"

export RESULT_PATH TASK_ID EPISODES
python - <<'PY'
import json
import os
import pathlib

path = pathlib.Path(os.environ["RESULT_PATH"])
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
task_id = int(os.environ["TASK_ID"])
episodes = int(os.environ["EPISODES"])
indices = [record["episode_idx"] for record in records]
if len(records) != episodes:
    raise SystemExit(f"expected {episodes} records, got {len(records)}")
if any(record["task_id"] != task_id for record in records):
    raise SystemExit("result contains an unexpected task id")
if sorted(indices) != list(range(episodes)):
    raise SystemExit("result episode indices are missing or duplicated")
successes = sum(bool(record["success"]) for record in records)
print(f"RESULT episodes={episodes} successes={successes} success_rate={successes / episodes:.4f}")
PY

echo "Evaluation records validated"
