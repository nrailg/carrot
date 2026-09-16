# Isaac uv 环境测试记录

## 测试思路

- 验证 `examples/isaac/pyproject.toml` 能在干净的 Gemini GPU 容器中解析、安装依赖并运行三个教学示例。
- 基础环境覆盖 Isaac Sim 和 Isaac Lab；`arena` 依赖组覆盖 Isaac Lab-Arena。
- 三个程序必须包含预期的仿真输出、没有 traceback 且退出码为 0，才判定通过。

## 测试代码

- `examples/isaac/pyproject.toml`：固定 Python、Isaac Lab、Isaac Sim、PyTorch 和 Arena 的兼容组合。
- `examples/isaac/README.md`：记录 `uv sync`、`uv run` 和可选 Arena 依赖组的运行方法。

## 执行步骤

```bash
cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot/examples/isaac
uv sync
uv run --no-group arena python 01_isaac_sim_falling_cube.py --headless
uv run --no-group arena python 02_isaac_lab_parallel_cartpole.py --num_envs 32 --num_steps 120 --viz none
uv sync --group arena
uv run --group arena python 03_isaac_lab_arena_composition.py
```

本轮实际使用的是 Gemini 容器 `mpi-launcher@mpi-1756637081-launcher`，宿主节点为
`mpi-1687844632-worker-2`，8 张 NVIDIA H20，驱动 `575.57.08`。在执行仿真前已完成
代码同步，并在 `/root/carrot-isaac-venv` 中执行冻结锁文件同步。

## 结果

- `uv sync --frozen --no-group arena`：通过。锁文件解析为 Isaac Lab
  `3.0.0b2.post1`、Isaac Sim `6.0.1.0`，并完成约 5.5 GiB 的增量安装。
- `01_isaac_sim_falling_cube.py`：未通过。Isaac Sim 启动后约 16 秒报告
  `VkResult: ERROR_DEVICE_LOST` / `A GPU crash occurred`，无 Python traceback。
- `02_isaac_lab_parallel_cartpole.py`：阻塞在 Cartpole USD 资产的 Nucleus
  `omni.client.stat` 检查，当前容器没有可用的本地资产缓存，未进入批量 step。
- `03_isaac_lab_arena_composition.py`：未继续执行；它依赖同一 Isaac Sim Vulkan
  初始化链路，不能把 01 的 GPU 失败伪装成 Arena 代码通过。

## 与历史基线的对照

- 2026-09-14，同一宿主节点、同样 H20/驱动，在旧的 `/root/conda` 环境中，01 的
  方块自由落体、02 的 32 个 Cartpole（动作 `(32, 1)`、120 步）和 03 的 Arena
  组合均通过。旧日志保存在 `docs/test-manifest/isaac-examples-20260914.artifacts/`。
- 因此本轮不能下结论说“换成 uv 后 Isaac 必然不兼容”。可以确认的是：uv 依赖安装
  成功；本轮节点随后出现的 Vulkan `ERROR_DEVICE_LOST` 与宿主 GPU 状态有关，尚未
  完成干净节点上的 uv 复验。
- 更早的 MuJoCo 历史复现另有明确边界：H20 + 575 同一张卡同时运行两个 classic
  MuJoCo EGL renderer 会触发 Xid 109；这是 MuJoCo/OpenGL graphics context 的
  capability 限制，不等同于 Isaac Lab 的 `num_envs` 批量环境，也不能据此声称
  CUDA 只允许一个进程占用 GPU。

## 当前结论

- 环境安装：`PASS`。
- Isaac 仿真运行：`BLOCKED`，当前节点在本轮多次 Isaac Sim 启动后出现新的
  `NVRM Xid 109 (CTX SWITCH TIMEOUT)`，需要 GPU reset/节点重启或换干净节点后再判定。
- 不应在该节点继续重复启动 Isaac Sim；重复启动可能把 H20 graphics/GSP 状态继续打坏。
