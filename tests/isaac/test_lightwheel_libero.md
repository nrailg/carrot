# Lightwheel-LIBERO 最小示例验证

目标：在独立 `/opt/venvs/lightwheel-libero` 中加载官方 LW-BenchHub 的
`L90K1PutTheBlackBowlOnThePlate`、Panda、单个 RGB 相机，运行 60 步。
保持末端位姿的示例策略不应被解释为已训练模型；成功率不用于模型比较。

## 运行

先确认 Gemini `MY_DFS`、代码同步和物理 GPU 的独占渲染；按需短暂暂停 dguard。
安装依赖与资产不属于测试脚本的隐式操作。

```bash
bash tests/isaac/test_lightwheel_libero.sh
```

验收：脚本退出码为 0；JSON 的任务、步数、环境数和成功信号类型符合预期；
RGB 为非空白 224×224 图像。主代理还需检查图像是否为预期场景。

## 2026-09-20

- 状态：PASS，单进程、单环境、60 步；`success_seen=[false]` 符合保持位姿策略的预期。
- GPU：Gemini H20，仅 GPU 0 用于本次仿真和渲染；运行前后恢复 dguard。
- LW-BenchHub：`b2bcb2d00edef691f9fcc49039cbf0bcc7464605`。
- 上游 Arena 子模块：`c7b70779f103e10d690d1a13863e8d77da7fc782`。
- 上游 Isaac Lab 子模块：`6acdd82a1633732d32bb575e3d792e34fdeb437e`。
- Carrot commit：`25c8cf06fc9c4d6eeb15d3378ed9e774f8be8ad8`，示例是工作区未提交文件。
- 独立 venv：Python 3.11.16、PyTorch 2.7.0+cu126、Isaac Sim 5.0.0.0、
  Isaac Lab 0.47.3、Arena 1.0.0、LW-BenchHub 0.1.0、Lightwheel SDK 1.0.1。
- SDK 固定 1.0.1 是必要的：上游导入 `lightwheel_sdk.loader.ENDPOINT`，
  SDK 1.0.2/1.0.3 已不再导出该名称。
- 证据目录：本次确认的 `${MY_DFS}/benchmarks/lightwheel-libero-20260920`。
- `smoke-camera-v2/result.json` 和 `camera.png`：脚本退出码 0；RGB 为 224×224，
  已人工复核黑碗、盘子与部分机器人可见。第一次相机取景只拍到墙面和台面，
  说明像素方差检查本身不足以证明任务物体入画。
- `batch4/result.json` 和 `camera.png`：同一进程、4 个环境、10 步，退出码 0；
  `success_seen` 长度为 4，首张 RGB 已人工复核碗盘入画。总用时 39 秒，包含启动和
  场景加载，不能视为稳态 FPS。两次运行后均无残留 Isaac 进程，dguard 已恢复。
- `uv pip check` 未全绿：Isaac Sim kernel 5.0.0.0 要求 Pillow 11.2.1，固定的
  Isaac Lab 0.47.3 和 IsaacLab-RL 0.4.4 要求 Pillow 11.3.0，无法同时满足；
  当前使用 11.3.0，两次真实 smoke 已通过。还有 packaging、click、
  typing-extensions、psutil、wheel 的版本提示，因此这套 venv 只验证了本例路径，
  不能视为上游所有功能的无冲突安装。
