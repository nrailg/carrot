#!/usr/bin/env bash
set -Eeuo pipefail

# 由当前 Gemini 会话预检确认路径和空闲 GPU；脚本不下载资产、不停止其他任务。
: "${MY_DFS:?Set MY_DFS from wx-detect-my-dfs}"
: "${ISAAC_ASSET_ROOT:?Set verified CephFS asset root containing Isaac/}"
: "${CUSTOM_ENV_RUN_DIR:?Set a new persistent result directory}"
: "${CUDA_VISIBLE_DEVICES:?Select a verified idle physical GPU}"
CARROT_DIR="${MY_DFS}/work/carrot"
source /opt/venvs/carrot/bin/activate
export PYTHONPATH="${CARROT_DIR}/src:${CARROT_DIR}/tests:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
test "$(command -v python)" = /opt/venvs/carrot/bin/python
test -f "$ISAAC_ASSET_ROOT/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
test -f "$ISAAC_ASSET_ROOT/Isaac/Environments/Grid/default_environment.usd"
test -f "$ISAAC_ASSET_ROOT/Isaac/Props/UIElements/frame_prim.usd"
cd "$CARROT_DIR"
mkdir -p "$CUSTOM_ENV_RUN_DIR"

# 串行启动，每个进程结束后才开始下一个，H20 不允许独立 renderer 共用同卡。
python examples/isaac_custom_env/run.py --viz none --num_envs 1 --steps 160 \
    --asset_root "$ISAAC_ASSET_ROOT"
python tests/isaac_custom_env/test_custom_env.py --viz none --asset_root "$ISAAC_ASSET_ROOT" \
    --output "$CUSTOM_ENV_RUN_DIR/lab.json"
python examples/isaac_custom_env/run.py --arena --viz none --num_envs 1 --steps 160 \
    --asset_root "$ISAAC_ASSET_ROOT"
python tests/isaac_custom_env/test_custom_env.py --arena --viz none --asset_root "$ISAAC_ASSET_ROOT" \
    --output "$CUSTOM_ENV_RUN_DIR/arena.json"
printf 'CUSTOM_ENV_MATRIX_PASS\n'
