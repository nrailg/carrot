# Isaac 本地资产测试记录

## 测试思路

- 验证 02 和 03 在设置 `ISAACSIM_ASSET_ROOT` 后只读取本地完整资产包并正常推进仿真。
- 验证无效资产根会在启动仿真前报告缺失目录。

## 测试代码

- `examples/isaac/asset_root.py`：检查资产根结构并覆盖 Isaac Lab 的资产路径常量。
- `examples/isaac/02_isaac_lab_parallel_cartpole.py`：在加载任务注册前应用本地资产根。
- `examples/isaac/03_isaac_lab_arena_composition.py`：在加载 Arena 资产前应用本地资产根。

## 执行步骤

```bash
python -m py_compile examples/isaac/asset_root.py \
  examples/isaac/02_isaac_lab_parallel_cartpole.py \
  examples/isaac/03_isaac_lab_arena_composition.py

python examples/isaac/02_isaac_lab_parallel_cartpole.py --num_envs 4 --num_steps 5 --viz none
python examples/isaac/03_isaac_lab_arena_composition.py --num_envs 1 --viz none
```

## 结果

- `PASS`：本地 `ruff`、`py_compile` 和 `git diff --check` 通过。
- `PASS`：Gemini 上 02 以 4 个环境推进 5 步并正常退出。
- `PASS`：Gemini 上 03 创建 1 个 Arena 环境、推进 60 步并正常退出。
- `PASS`：无效资产根在启动仿真前报告缺失 `Isaac` 和 `NVIDIA` 目录。
- `PASS`：清空三个环境变量后，代码内默认 EULA、资产根和 GPU 配置使 02/03 正常运行。
