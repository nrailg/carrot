#!/usr/bin/env bash
set -euo pipefail

: "${DFS:?请设置个人 DFS 目录}"
export DISPLAY="${DISPLAY:-:99}"
NOVNC_PORT="${NOVNC_PORT:-8080}"
VNC_PORT="${VNC_PORT:-5900}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM="${1:-$SCRIPT_DIR/spawn_prims.py}"
if (( $# > 0 )); then shift; fi
[[ -f "$PROGRAM" ]] || { echo "Python 文件不存在：$PROGRAM" >&2; exit 1; }
for dependency in Xvfb xdpyinfo openbox x11vnc websockify; do
    command -v "$dependency" >/dev/null || { echo "缺少依赖：$dependency" >&2; exit 1; }
done
[[ -f /usr/share/novnc/vnc.html ]] || { echo "缺少 /usr/share/novnc/vnc.html" >&2; exit 1; }
[[ ! -e "/tmp/.X${DISPLAY#:}-lock" ]] || { echo "DISPLAY 已占用：$DISPLAY" >&2; exit 1; }
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then kill "$pid"; fi
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
Xvfb "$DISPLAY" -screen 0 1280x800x24 -nolisten tcp &
pids+=("$!")
ready=0
for _ in {1..50}; do
    if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.1
done
[[ "$ready" == 1 ]] || { echo "Xvfb 启动失败" >&2; exit 1; }
openbox &
pids+=("$!")
x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared \
    -nopw -noxdamage -noxrecord -noscr -threads &
pids+=("$!")
/usr/bin/websockify --web=/usr/share/novnc "${NOVNC_BIND:-0.0.0.0}:$NOVNC_PORT" \
    "127.0.0.1:$VNC_PORT" &
pids+=("$!")

source /opt/venvs/carrot/bin/activate
export PYTHONPATH="$SCRIPT_DIR/../../src:$SCRIPT_DIR/../../tests:${PYTHONPATH:-}"
export ISAACSIM_ASSET_ROOT="${DFS%/}/isaacsim_assets/Assets/Isaac/6.0"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset CUDA_VISIBLE_DEVICES
echo "noVNC: http://${__HOST_IP__:-localhost}:$NOVNC_PORT/vnc.html"
taskset -c "${ISAAC_CPUSET:-0-15}" python "$PROGRAM" --viz kit \
    --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=16' "$@" &
pids+=("$!")
wait -n "${pids[@]}"
