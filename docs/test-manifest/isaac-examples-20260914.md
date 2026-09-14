# Isaac 入门示例测试记录

## 测试思路

- 目标：在用户提供的 Gemini GPU 容器中运行 `examples/isaac/` 下的三个示例，验证示例与容器中的 Isaac Sim、Isaac Lab 和 Isaac Lab-Arena API 是否匹配。
- 覆盖的行为与不覆盖的边界：覆盖模块导入、仿真启动、场景构建、批量环境 step 和正常退出；不评估物理精度、渲染质量、训练收敛或性能上限。
- 预期通过/失败条件：示例在对应运行环境中完成且退出码为 0 时通过；导入错误、API 不兼容、运行时异常、超时或非零退出码均不通过。容器未安装某一组件时记为 `BLOCKED`，不视作代码通过。

## 测试代码

- 新增或修改的测试：无 pytest 用例；本次直接运行三个教学示例。
- 被测代码：`examples/isaac/01_isaac_sim_falling_cube.py`、`examples/isaac/02_isaac_lab_parallel_cartpole.py`、`examples/isaac/03_isaac_lab_arena_composition.py`。
- 相关改动/提交：工作树中新增的 `examples/isaac/`，尚未提交。

## 执行步骤

1. 代码版本与同步检查：将 `examples/isaac/` 同步到 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot/examples/isaac/`，并用 `cmp` 核对四个文件。全仓 `rsync -a` 因 CephFS 不允许复制属组且远端已有 `.codex` 链接而返回 23，随后改为只同步本次文件并使用 `--no-owner --no-group`，同步成功。
2. 环境、Ray 与依赖准备：Gemini 容器 `mpi-launcher@mpi-1742693300-launcher`，单节点 8 张 NVIDIA H20；测试使用 `CUDA_VISIBLE_DEVICES=0`。容器没有 dguard，也没有预装 Isaac 包；从共享盘 wheel 安装 Isaac Sim 6.0.1.0，使用 Arena `release/0.3.0` 自带的 Isaac Lab 3.0.0 submodule，并通过代理安装 Arena Python 依赖。示例不使用 Carrot distributed API，因此未启动 Ray。
3. 精确测试命令：

   ```bash
   python examples/isaac/01_isaac_sim_falling_cube.py --headless
   python examples/isaac/02_isaac_lab_parallel_cartpole.py \
     --num_envs 32 --num_steps 120 --viz none
   python examples/isaac/03_isaac_lab_arena_composition.py
   ```

   02 和 03 运行前将对应 Isaac Lab `source/*` 目录加入 `PYTHONPATH`；03 还加入临时检出的 `/root/IsaacLab-Arena`。Arena 兼容版本为提交 `3032c7888b7b56e54d907e5c46e98c6dfdfb7dba`，其 Isaac Lab submodule 为 `af1bab4dc173ba69b08fab779c14ead61d13fd33`。

## 实际结果

- 状态：`PASS`
- 每次运行：
  - 2026-09-14 19:37 CST，01 运行 16 秒，退出码 0。方块从约 0.973 m 落到 0.100 m 并稳定在地面上，300 帧后正常关闭。
  - 2026-09-14 19:43 CST，02 第一次运行出现 `kit` visualizer 与 headless 配置冲突；外层启动器错误返回 0，但日志含 traceback，因此判为失败。移除默认 `kit` visualizer 后重跑。
  - 2026-09-14 19:44 CST，02 运行 20 秒，退出码 0。建立 32 个 replicated physics 环境，动作张量形状 `(32, 1)`，完成 120 个批量 step 后正常关闭。
  - 2026-09-14 19:50--19:53 CST，03 先后发现 Arena `main` 与 Sim 6.0 版本不匹配，以及缺少 Pinocchio、ONNX/RSL-RL、HDF5 依赖；切换至官方兼容的 `release/0.3.0` 并经代理补齐依赖。
  - 2026-09-14 19:54 CST，03 带显式 headless 参数运行 46 秒，退出码 0。完成厨房、Franka、两个物体的组合，动作空间为 `(1, 7)`，完成 60 个零动作 step 后正常关闭。
  - 2026-09-14 19:54 CST，03 按 README 的无参数命令复跑 19 秒，退出码 0，结果相同。
- 失败、阻塞或偏差及判断：Arena 官方厨房资产会报告细长把手无法生成 GPU-compatible convex hull、部分刚体惯量不完整等 PhysX warning；仿真会按官方资产配置回退处理，未影响环境建立、step 或退出。Arena `main` 已转向更新的依赖组合，不能与本容器的 Isaac Sim 6.0.1 混用；示例按 `release/0.3.0` 的兼容矩阵验证。
- 证据：`docs/test-manifest/isaac-examples-20260914.artifacts/`，其中 `01-isaac-sim.log`、`02-isaac-lab.log` 和 `03-isaac-arena-readme-command.log` 是最终通过日志；带 `attempt`、`mismatch` 或 `missing` 的文件保留修复前失败过程。
- 清理动作与残留：所有后台测试任务均已结束并导出日志；没有启动 Ray。Isaac 依赖和 `/root/IsaacLab-Arena` 仅安装在本次临时容器中，未写入 Carrot 依赖配置。

## 后续

- 需要修复、重跑或由用户决定的事项：无。若未来升级到 Isaac Sim 6.1，应同时升级 Arena 和它固定的 Isaac Lab submodule，再重新验证 03。
