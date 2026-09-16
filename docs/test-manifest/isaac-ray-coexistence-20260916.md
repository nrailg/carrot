# Isaac 与 Ray 共存测试记录

## 测试思路

- 仅通过 `source /opt/venvs/carrot/bin/activate` 激活镜像环境，确认解释器和核心依赖来自该 venv。
- 运行现有 Ray runtime 单测与最小 actor smoke，验证 Ray 能正常启动、调度和关闭。
- 在 Ray 完整关闭后串行运行 Isaac 示例，检查导入、GPU 仿真、退出和残留进程，避免 H20 上多个 renderer 共享同一张 GPU。

## 测试代码

- `tests/test_ray_runtime.py`：验证 Carrot 的 Ray worker、placement、channel 和环境变量传播。
- `examples/isaac/01_isaac_sim_falling_cube.py`：验证 Isaac Sim 的基础启动、物理步进与正常退出。
- `examples/isaac/02_isaac_lab_parallel_cartpole.py`：验证单 renderer 内的 Isaac Lab 批量环境。

## 执行步骤

```bash
source /opt/venvs/carrot/bin/activate
cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
pytest -v -s --timeout=1800 tests/test_ray_runtime.py

cd examples/isaac
CUDA_VISIBLE_DEVICES=0 python 01_isaac_sim_falling_cube.py --headless
CUDA_VISIBLE_DEVICES=0 python 02_isaac_lab_parallel_cartpole.py \
  --num_envs 4 --num_steps 10 --device cuda:0 --viz none
```

## 结果

- `NOT RUN`：等待远程 Gemini 环境执行。
