# SO101 左右转动

先读 `run_isaac_sim.py`：启动应用 → 创建地面和灯光 → 加载机器人 → 初始化关节 → 循环发送目标。
只有一个机器人，`shoulder_pan` 在 ±30° 之间按正弦曲线往复，完整周期为 4 秒仿真时间；
其他五个关节保持零位。通过位置驱动器产生运动，只有初始化时直接设置关节位置。

使用 Isaac Sim 6.0 环境及官方 **PhysX** 资产：
`Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd`。
保留整个资产目录中的 configuration、mesh 和材质引用，不能只复制顶层 USD。
程序只读取明确指定的本地资产，不在启动时下载。

```bash
source /opt/venvs/carrot/bin/activate
python examples/isaac_so101/run_isaac_sim.py \
  --robot-usd /absolute/path/to/Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd
```

默认一直运行，关闭窗口或 Ctrl-C 退出。`--amplitude 20 --period 6` 改为 ±20°、6秒一周期。
`--gpu 1` 选择第1号GPU。没有图形桌面时使用 `--headless --steps 960` 跑两个默认周期；
添加 `--trace /absolute/path/to/motion.npz` 保存目标、实测关节角及基座位姿供检查。
GUI 暂停时不推进目标；重新运行程序从零位开始。

若容器同时存在 `/etc/vulkan/icd.d/nvidia_icd.json` 和
`/usr/share/vulkan/icd.d/nvidia_icd.json`，且 Kit 报告重复 NVIDIA ICD，运行前显式选择一份：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

本次测试容器需要此设置，否则 Vulkan 会把每张显卡枚举两次，`--gpu 1` 会选到物理 GPU 0。

底座固定关节、质量、惯量、碰撞和力矩上限来自 USD；脚本显式设置位置 PD 增益，
关闭自碰撞，保留重力。这是基础运动演示，不是 SO101 真机动力学标定。

参考：[官方机器人资产目录](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_robots.html#robotstudio)、
[官方 SO101 配置和高增益参数](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/robots/so101.py)。
本例直接使用 Isaac Sim Core API，不依赖较新 Isaac Lab 的 SO101 配置导入。

验证入口：`tests/isaac_so101/test_motion.sh`，实际验证状态见同目录 `test_motion.md`。
2026-10-26 在 Isaac Sim 6.0.1.0 / RTX PRO 5000 上通过两个周期的物理测试：
实测转角约 ±28.62°，基座位移为0。目标与实测存在 PD 跟踪滞后，本例不要求精确轨迹复现。
