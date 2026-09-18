#!/usr/bin/env bash

set -Eeuo pipefail

CARROT_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot"
OPENPI_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi"
CHECKPOINT_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_libero_pytorch"
TOKENIZER_DIR="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/google/paligemma-3b-pt-224"
OUTPUT_ROOT="/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-smoke"
CARROT_PYTHON="/opt/venvs/carrot/bin/python"
LIBERO_PYTHON="/opt/venvs/openpi-libero/bin/python"
DGUARD="/root/dguard/dguard.sh"

POLICY_GPU=0
RENDER_GPU=1
PORT=8000
DGUARD_STOP_MINUTES=30
RUN_DIR="${OUTPUT_ROOT}/$(date +%Y%m%d-%H%M%S)"
RENDER_LOG="${RUN_DIR}/render-smoke.log"
SERVER_LOG="${RUN_DIR}/server.log"
EVAL_LOG="${RUN_DIR}/eval.log"
RESULT_PATH="${RUN_DIR}/episodes.jsonl"
VIDEO_DIR="${RUN_DIR}/videos"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

gpu_pids() {
    nvidia-smi -i "$1" -q -d PIDS 2>/dev/null \
        | awk -F: '/Process ID/{gsub(/ /, "", $2); print $2}' \
        | sort -u
}

wait_for_gpu_to_clear() {
    local gpu="$1"
    local pids=""
    for _ in $(seq 1 30); do
        pids="$(gpu_pids "$gpu")"
        [[ -z "$pids" ]] && return
        sleep 1
    done
    die "physical GPU ${gpu} is occupied by PID(s): ${pids}"
}

assert_egl_device_mapping() {
    PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src:/opt/libero/src:${PYTHONPATH:-}" \
    PYOPENGL_PLATFORM=egl \
    "$LIBERO_PYTHON" - "$RENDER_GPU" <<'PY'
import ctypes
import sys

from OpenGL import EGL
from mujoco.egl import egl_ext

query_attrib_type = ctypes.CFUNCTYPE(
    EGL.EGLBoolean,
    EGL.EGLDeviceEXT,
    EGL.EGLint,
    ctypes.POINTER(ctypes.c_ssize_t),
)
query_attrib = query_attrib_type(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
device_index = int(sys.argv[1])
devices = egl_ext.eglQueryDevicesEXT()
cuda_device = ctypes.c_ssize_t(-1)
assert query_attrib(devices[device_index], 0x323A, ctypes.byref(cuda_device))
assert cuda_device.value == device_index, (device_index, cuda_device.value)
print(f"EGL_MAPPING_OK egl_device={device_index} cuda_device={cuda_device.value}")
PY
}

SERVER_PID=""
RENDER_PID=""
DGUARD_PAUSED=0
XID_COUNT_BEFORE=""

cleanup() {
    local rc=$?
    local xid_count_after=""
    trap - EXIT INT TERM
    if [[ -n "$RENDER_PID" ]]; then
        kill "$RENDER_PID" 2>/dev/null || true
        wait "$RENDER_PID" 2>/dev/null || true
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
                | tee "${RUN_DIR}/new-xid.log" || true
            bash "$DGUARD" off || true
            rc=1
        else
            bash "$DGUARD" on || true
        fi
    fi
    echo "artifacts: ${RUN_DIR}"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$POLICY_GPU" != "$RENDER_GPU" ]] || die "policy and renderer must use different GPUs"
[[ -x "$CARROT_PYTHON" ]] || die "missing Carrot Python: ${CARROT_PYTHON}"
[[ -x "$LIBERO_PYTHON" ]] || die "missing LIBERO Python: ${LIBERO_PYTHON}"
[[ -f "$DGUARD" ]] || die "missing dguard: ${DGUARD}"
[[ -f "${OPENPI_DIR}/examples/libero/main.py" ]] || die "missing OpenPI evaluator"
grep -q "task_id" "${OPENPI_DIR}/examples/libero/main.py" || \
    die "OpenPI evaluator does not support --args.task-id"
grep -q "result_out_path" "${OPENPI_DIR}/examples/libero/main.py" || \
    die "OpenPI evaluator does not support durable JSONL results"
[[ -s "${CHECKPOINT_DIR}/model.safetensors" ]] || die "missing checkpoint weights"
[[ -f "${CHECKPOINT_DIR}/config.json" ]] || die "missing checkpoint config"
[[ -f "${CHECKPOINT_DIR}/assets/physical-intelligence/libero/norm_stats.json" ]] || \
    die "missing LIBERO normalization stats"
[[ -f "${TOKENIZER_DIR}/tokenizer_config.json" ]] || die "missing HuggingFace tokenizer"

cd "$CARROT_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${OPENPI_DIR}/packages/openpi-client/src:/opt/libero/src:${PYTHONPATH:-}"

echo "[1/3] CPU contract regression"
"$CARROT_PYTHON" -m ruff check \
    src/carrot/models/pi05/transforms.py \
    src/carrot/models/pi05/preprocessing.py \
    src/carrot/models/pi05/embodiments \
    src/carrot/models/pi05/inference/policy.py \
    src/carrot/models/pi05/inference/policy_config.py \
    src/carrot/models/pi05/inference/__init__.py \
    src/carrot/cli/serve_pi05_policy.py \
    tests/test_pi05_inference.py \
    tests/test_pi05_inference_checkpoint.py \
    tests/test_pi05_modeling.py
"$CARROT_PYTHON" -m pytest -q --timeout=1800 \
    tests/test_pi05_modeling.py \
    tests/test_pi05_inference.py \
    tests/test_pi05_inference_checkpoint.py

command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
GPU_COUNT="$(nvidia-smi --list-gpus | wc -l)"
[[ "$POLICY_GPU" -lt "$GPU_COUNT" ]] || die "policy GPU is out of range"
[[ "$RENDER_GPU" -lt "$GPU_COUNT" ]] || die "renderer GPU is out of range"
mkdir -p "$RUN_DIR" "$VIDEO_DIR"

bash "$DGUARD" stop "$DGUARD_STOP_MINUTES"
DGUARD_PAUSED=1
wait_for_gpu_to_clear "$POLICY_GPU"
wait_for_gpu_to_clear "$RENDER_GPU"
if dmesg >/dev/null 2>&1; then
    XID_COUNT_BEFORE="$(dmesg | grep -ci 'NVRM: Xid' || true)"
fi

export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
assert_egl_device_mapping

echo "[2/3] one-process LIBERO EGL smoke on physical GPU ${RENDER_GPU}"
READY_PATH="${RUN_DIR}/render.ready"
export READY_PATH
CUDA_VISIBLE_DEVICES="$RENDER_GPU" \
MUJOCO_EGL_DEVICE_ID="$RENDER_GPU" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
"$LIBERO_PYTHON" -u - <<'PY' >"$RENDER_LOG" 2>&1 &
import os
import pathlib
import time

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    camera_heights=256,
    camera_widths=256,
)
try:
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = obs[key]
        assert image.shape == (256, 256, 3), (key, image.shape)
        assert image.dtype == np.uint8, (key, image.dtype)
        assert float(image.std()) > 1.0, (key, float(image.std()))
    print(f"RENDER_OK pid={os.getpid()} agentview_std={obs['agentview_image'].std():.3f}")
    pathlib.Path(os.environ["READY_PATH"]).touch()
    time.sleep(10)
finally:
    env.close()
PY
RENDER_PID=$!

for _ in $(seq 1 120); do
    [[ -f "$READY_PATH" ]] && break
    kill -0 "$RENDER_PID" 2>/dev/null || {
        sed -n '1,240p' "$RENDER_LOG" >&2
        die "LIBERO renderer smoke exited before becoming ready"
    }
    sleep 1
done
[[ -f "$READY_PATH" ]] || die "LIBERO renderer smoke did not become ready"
if ! wait "$RENDER_PID"; then
    sed -n '1,240p' "$RENDER_LOG" >&2
    die "LIBERO renderer smoke failed"
fi
RENDER_PID=""
sed -n '1,240p' "$RENDER_LOG"
wait_for_gpu_to_clear "$RENDER_GPU"

echo "[3/3] Carrot PI0.5 server plus one official LIBERO episode"
if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
    die "port ${PORT} is already in use"
fi
CUDA_VISIBLE_DEVICES="$POLICY_GPU" \
"$CARROT_PYTHON" -u -m carrot.cli.serve_pi05_policy \
    --embodiment libero \
    --checkpoint "$CHECKPOINT_DIR" \
    --tokenizer-path "$TOKENIZER_DIR" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for second in $(seq 1 900); do
    kill -0 "$SERVER_PID" 2>/dev/null || {
        sed -n '1,240p' "$SERVER_LOG" >&2
        die "policy server exited before becoming ready"
    }
    if "$CARROT_PYTHON" - "$PORT" <<'PY' 2>/dev/null
import sys
import urllib.request

with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=1) as response:
    assert response.read() == b"OK\n"
PY
    then
        ready=1
        break
    fi
    if (( second % 30 == 0 )); then
        echo "waiting for policy server: ${second}s/900s"
    fi
    sleep 1
done
[[ "$ready" -eq 1 ]] || die "policy server did not become ready"
policy_pids="$(gpu_pids "$POLICY_GPU")"
[[ "$(wc -w <<<"$policy_pids")" -eq 1 ]] || \
    die "expected one process on policy GPU ${POLICY_GPU}, found: ${policy_pids:-none}"
render_pids="$(gpu_pids "$RENDER_GPU")"
[[ -z "$render_pids" ]] || \
    die "renderer GPU ${RENDER_GPU} became occupied before evaluation: ${render_pids}"
echo "policy server uses physical GPU ${POLICY_GPU}, host PID ${policy_pids}"

CUDA_VISIBLE_DEVICES="$RENDER_GPU" \
MUJOCO_EGL_DEVICE_ID="$RENDER_GPU" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
"$LIBERO_PYTHON" -u "${OPENPI_DIR}/examples/libero/main.py" \
    --args.host=127.0.0.1 \
    --args.port="$PORT" \
    --args.task-suite-name=libero_spatial \
    --args.task-id=0 \
    --args.num-trials-per-task=1 \
    --args.replan-steps=5 \
    --args.seed=7 \
    --args.video-out-path="$VIDEO_DIR" \
    --args.result-out-path="$RESULT_PATH" \
    >"$EVAL_LOG" 2>&1 &
RENDER_PID=$!
if ! wait "$RENDER_PID"; then
    sed -n '1,320p' "$EVAL_LOG" >&2
    die "LIBERO evaluator failed"
fi
RENDER_PID=""
sed -n '1,320p' "$EVAL_LOG"

"$CARROT_PYTHON" - "$RESULT_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert len(records) == 1, records
assert records[0]["task_id"] == 0, records[0]
assert records[0]["episode_idx"] == 0, records[0]
print(f"E2E_OK success={bool(records[0]['success'])} result={path}")
PY

echo "PASS: PI0.5 LIBERO inference and rendering smoke"
