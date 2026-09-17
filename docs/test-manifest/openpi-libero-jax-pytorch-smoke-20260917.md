# OpenPI LIBERO JAX / PyTorch 快速闭环测试记录

## 测试思路

- 使用官方 `pi05_libero` checkpoint 和 checkpoint 自带的 normalization stats。
- 顺序运行 JAX、PyTorch 两个后端；统一使用 `libero_spatial`、seed 7、replan steps 5。
- 每个 task 只运行 1 个 episode，共 10 episodes/backend；该结果仅用于快速闭环 smoke 和粗略对比。
- 每次只启动一个 LIBERO EGL renderer，并与 policy server 使用不同 GPU。

## 测试代码

- `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi/examples/libero/main.py`：官方 LIBERO observation、action replanning、初始状态和成功判定。
- `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi/scripts/serve_policy.py`：根据 checkpoint 格式分别加载 JAX 或 PyTorch policy。

## 执行步骤

```bash
OPENPI=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi
JAX_CKPT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_libero
OUT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-jax-pytorch-smoke-20260917
PT_CKPT=/root/pi05_libero_pytorch

source /opt/venvs/openpi-libero/bin/activate
bash /root/dguard/dguard.sh status --local
bash /root/dguard/dguard.sh stop 120
mkdir -p "$OUT"/jax/{logs,results,videos} "$OUT"/pytorch/{logs,results,videos}

export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
```

JAX server 在 GPU 0 作为远程 background task 启动：

```bash
source /opt/venvs/openpi-libero/bin/activate
export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
export CUDA_VISIBLE_DEVICES=0
cd "$OPENPI"
exec python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$JAX_CKPT"
```

server ready 后，client 在 GPU 1 运行唯一的 EGL renderer：

```bash
source /opt/venvs/openpi-libero/bin/activate
export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
export CUDA_VISIBLE_DEVICES=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$OPENPI"
exec python examples/libero/main.py \
  --args.host=127.0.0.1 \
  --args.port=8000 \
  --args.task-suite-name=libero_spatial \
  --args.num-trials-per-task=1 \
  --args.replan-steps=5 \
  --args.seed=7 \
  --args.video-out-path="$OUT/jax/videos" \
  --args.result-out-path="$OUT/jax/results/episodes.jsonl"
```

JAX client 完成后，停止 JAX server，再转换同一 checkpoint：

```bash
source /opt/venvs/openpi-libero/bin/activate
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
cd "$OPENPI"
exec python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir="$JAX_CKPT" \
  --config-name=pi05_libero \
  --output-path="$PT_CKPT" \
  --precision=bfloat16

test ! -e "$PT_CKPT/assets"
cp -r "$JAX_CKPT/assets" "$PT_CKPT/assets"
```

PyTorch 默认 compile server 使用与 JAX server 相同的命令，只把 checkpoint 改为：

```bash
source /opt/venvs/openpi-libero/bin/activate
export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
export CUDA_VISIBLE_DEVICES=0
cd "$OPENPI"
exec python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$PT_CKPT"
```

PyTorch client：

```bash
source /opt/venvs/openpi-libero/bin/activate
export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
export CUDA_VISIBLE_DEVICES=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$OPENPI"
exec python examples/libero/main.py \
  --args.host=127.0.0.1 \
  --args.port=8000 \
  --args.task-suite-name=libero_spatial \
  --args.num-trials-per-task=1 \
  --args.replan-steps=5 \
  --args.seed=7 \
  --args.video-out-path="$OUT/pytorch/videos" \
  --args.result-out-path="$OUT/pytorch/results/episodes.jsonl"
```

默认 `torch.compile(mode="max-autotune")` 首次编译导致 websocket timeout 后，停止
server，以不修改源码的运行时 eager 覆盖重新启动：

```bash
source /opt/venvs/openpi-libero/bin/activate
export LIBERO_CONFIG_PATH=/opt/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src:/opt/libero/src"
export CUDA_VISIBLE_DEVICES=0
cd "$OPENPI"
python - <<'PY'
import runpy
import sys
import torch

sys.argv = [
    "scripts/serve_policy.py",
    "--port=8000",
    "policy:checkpoint",
    "--policy.config=pi05_libero",
    "--policy.dir=/root/pi05_libero_pytorch",
]
torch.compile = lambda fn, *args, **kwargs: fn
runpy.run_path("scripts/serve_policy.py", run_name="__main__")
PY
```

两个 server 均通过远程 task stop 停止；使用 task dump 将完整输出分别保存到
`$OUT/jax/logs/` 和 `$OUT/pytorch/logs/`。

## 安全 runner 10-episode 回归

### 测试思路

- 使用 `examples/libero/run_openpi_libero_eval.sh` 运行 PyTorch eager policy。
- 仅运行 `libero_spatial` task 0 的 10 个 episodes；policy 固定使用 GPU 0，唯一的 EGL
  renderer 固定使用 GPU 1。
- 验证 tokenizer 代理预取、server 就绪检查、JSONL 完整性、温度记录、Xid 检查以及
  server/dguard 清理路径。

### 测试代码

- `examples/libero/run_openpi_libero_eval.sh`：拒绝同卡 server/renderer；结果必须恰好包含
  task 0 的 episode 0–9；出现新增 Xid 时返回失败并保持 dguard 关闭。

### 执行步骤

```bash
cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
./examples/libero/run_openpi_libero_eval.sh --episodes 10
```

### 结果

- `FAIL`（首次）：tokenizer 代理预取成功，server 正常就绪；renderer 在 import 阶段被
  robosuite 拒绝，因为 `MUJOCO_EGL_DEVICE_ID=0` 不在 `CUDA_VISIBLE_DEVICES=1` 中。未执行
  episode、未新增 Xid，server 已停止且 dguard 已恢复。runner 已改为传 renderer 的物理
  GPU 编号，并将裸 TCP 探活改为合法 websocket 握手。
- `FAIL`（修复后重跑）：完整生成 task 0 的 episode 0–9，结果为 8/10 success；JSONL、
  10 个独立视频和日志均已持久化。但物理 GPU 1（PCI `3a:00.0`）在 rollout 中新增多次
  Xid 31，并出现 Xid 109 `CTX SWITCH TIMEOUT`，因此 runner 正确返回 exit code 1。
- 运行期间 GPU 0/1 的核心最高温度分别为 35°C/38°C，显存最高温度为 37°C/39°C，
  不属于过热。测试结束后无 policy/evaluator GPU 进程；检测到新 Xid 后 runner 保持 dguard
  关闭，未自动换卡或重试。
- 失败证据：
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/libero_spatial-task0-10x-20260917-114020`。
- `PASS`（GPU 2 重跑）：仅将 renderer 从物理 GPU 1 改到物理 GPU 2，保持 GPU 0 policy、
  task 0、10 episodes 和其他参数不变。完整生成 episode 0–9，10/10 success，runner exit
  code 0，Xid 计数保持为 12、没有新增记录。
- GPU 0/2 的核心最高温度分别为 35°C/32°C，显存最高温度为 36°C/33°C。测试结束后
  policy/evaluator 均已停止，dguard 正常恢复。runner 默认 renderer 已改为验证通过的 GPU 2。
- 通过证据：
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/libero_spatial-task0-10x-20260917-115023`。

## 结果

- `PASS`：JAX 完成 10 个唯一 `(task_id, episode_idx)`，10/10 success，exit code 0，
  evaluator 用时约 121 秒。
- `PASS`：PyTorch eager 完成同一组 10 个 episode，10/10 success，exit code 0，
  evaluator 用时约 149 秒。
- PyTorch checkpoint 由同一 JAX checkpoint 转换，`model.safetensors` 为 7,233,650,408 bytes；
  normalization stats 从原 checkpoint assets 复制。
- `WARN`：配置默认的 `torch.compile(mode="max-autotune")` 首次编译超过 websocket keepalive，
  10 个请求均因连接超时失败。保留该次 0/10 诊断记录；随后仅在运行时禁用 compile，
  未修改源码，以 eager 路径得到上述正式 10/10 smoke 结果。
- 持久化结果：
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-jax-pytorch-smoke-20260917`。
- 测试结束后 policy server/client 均已停止，GPU 无残留进程；dguard 已安排自动恢复。
