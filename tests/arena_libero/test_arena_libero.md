# Arena LIBERO 单任务 RL 验证

状态：PASS（2026-09-22 ray-04）。CPU 11 项、真实 GPU 控制 / PPO / 相机及 Ray actor 验证通过。

目标：从原生 Arena 场景、机器人、任务配置构建黑碗放盘子环境，不导入 LW-BenchHub。
资产可复用已有 USD；本任务不是原始 MuJoCo LIBERO 完整 benchmark 的等价实现。

依赖：以 pyproject.toml / uv.lock 为准，Python 3.12、Isaac Sim 6.0.1.0、
Isaac Lab 3.0.0b2.post1、Arena commit 8737b4ceb25f99f81a81786b7fde73139b52f324。
Docker image tag：未记录（运行容器中版本文件不存在）。Carrot 基线：ed93c5f（nrwu/newLiberoWithArena）。

预期：单个 H20 renderer / 4 environments，两个 RGB 相机、7 维动作，
reset 子集隔离、终止和超时语义、terminal observation、PPO 参数更新和 Ray worker。
CPU 测试检验快照所有权、bootstrap mask、输入约束；真实 GPU 测试必须独立通过。

最终结果见 ray-04；此前失败记录保留用于说明修复和验收边界。

## 执行方式

先检查并暂停 dguard，确认 GPU 0 没有其他 renderer。显式资产目录包含三个入口：

- `scene.usd` → SDK 缓存的 `floorplan/robocasa-libero-1-1/scene_enabled.usd`
- `bowl.usd` → `object/Bowl008/Bowl008.usd`
- `plate.usd` → `object/Plate012/Plate012.usd`

不修改缓存 USD；`asset()` 会解析链接，以保留 USD 的相对引用关系。
Panda 默认引用 NVIDIA Isaac 5.1 资产 URL；这只是机器人资产版本，运行时仍是 Isaac Sim 6。
可通过 `ArenaLiberoConfig.panda_usd` 提供离线本地副本。

```bash
export ARENA_LIBERO_ASSETS=/root/arena-libero-assets
export ARENA_LIBERO_OUTPUT="${MY_DFS}/benchmarks/arena-libero-rl/<new-run>"
bash tests/arena_libero/test_arena_libero.sh
```

`run_arena_libero.py` 验证两个相机、部分 reset 隔离、terminal observation、真实接触成功状态，
然后采样 4×32 transitions 做两次 PPO 参数更新。PPO actor 只用 8 维机器人状态；
critic 用 52 维特权状态。这是训练链路 smoke，不是成功率或视觉策略学习验收。

同环境 Ray 验证：激活 `/opt/venvs/carrot`，设置本项目 `PYTHONPATH` 和现有集群的
`RAY_ADDRESS`。runner 给 actor 显式传递源码路径和环境，并预留完整 1 GPU / 2 CPU。
本 H20 验收入口要求分配到 GPU 0，否则在启动 Kit 前失败。不要停止或清理无关集群。

```bash
python tests/arena_libero/run_ray.py --asset-root "$ARENA_LIBERO_ASSETS" \
  --output "${MY_DFS}/benchmarks/arena-libero-rl/<new-ray-run>"
```

驱动端只使用 Carrot Cluster/Worker API；actor 内启动 Kit，完成相同验收后再次调用 actor，
检查 Python/Ray 版本、解释器和 PID 一致，退出后写 `ray_result.json`。


## 2026-09-22 native-01

远端 Python 3.12.13 / Torch 2.11.0+cu128 / Ray 2.58.0，Isaac 组件与上述锁一致。
任务 ID：`5f3f1fce-0640`。输出：`${MY_DFS}/benchmarks/arena-libero-rl/20260922-native-01`。
资产 root：`/root/arena-libero-assets`，链接解析至 `/root/.cache/lightwheel_sdk` 的上述三个 USD。
场景单位为米，Z-up；碗 bounds 尺寸约 0.1803×0.1803×0.0824 m，
盘 bounds 尺寸约 0.2746×0.2740×0.0299 m。均使用原始缩放 1。
厨房前台面约 z=0.75m，Panda 固定在 (2.432,-1.581,0.75)m，朝 -Y。
仅读取和复用 USD，不导入 LW-BenchHub Python 包。


native-01 结果：CPU 11 passed（1.78s）；启动参数 `--kit_args` 的值以 `--` 开头，
argparse 误认成另一个参数，未启动 Kit。已改为 `--kit_args=VALUE`，重跑 native-02。


## 2026-09-22 native-02

任务 ID：`5f3f1fce-0645`，产物目录：`${MY_DFS}/benchmarks/arena-libero-rl/20260922-native-02`。
CPU 11 项通过。仿真断言通过：4 env / 128 transitions / 8 truncations，
actor 最大更新 0.0006003901，critic 最大更新 0.0005998611，loss=-0.00850318。
真实接触成功 fixture、partial reset、terminal observation PASS，`lw_imported=false`。

整轮状态：FAIL。`SimulationContext cleared` 后进程退出码 139；外部相机碗盘可见，
腕部相机主要拍到机械臂，视觉验收未通过。JSON 内的 PASS 仅代表仿真断言，不代表进程退出或视觉验收。
已恢复 dguard，确认无遗留 renderer。下一轮使用 AppLauncher 默认 fast shutdown，
增加相机位姿记录、reset 重渲染与 5 个中性动作预热步，继续定位腕部视角。

## 2026-09-22 native-03

任务 ID：`5f3f1fce-0657`。完整产物目录：
`${MY_DFS}/benchmarks/arena-libero-rl/20260922-native-03`。
进程 exit 0，仿真断言 PASS，4 env / 128 transitions / 8 truncations；
actor 最大更新 0.0006003827，critic 最大更新 0.0005998462。
主模型独立读取 result.json 并查看两路图像：预热后碗、盘子和夹爪可见。

`camera_poses.json` 中 camera 数据是传感器报告的 world pose，不能直接当成当前物理 hand
的刚性外参：其位置与 hand 的距离不等于配置的 mount offset。图像内容已跟随机器人姿态更新；
下一轮另外保留 reset 首帧，用于检查视觉观测时序。控制和任务判定只使用 PhysX 物体/机器人状态。

关闭方式改回 AppLauncher 默认 fast shutdown；没有修改第三方源码。
Ray worker 的环境先显式 close，Kit 的进程生命周期由 Carrot Cluster 终止专用 actor 管理；
不在需要返回 RPC 的 actor teardown 中调用会退出整个进程的 `SimulationApp.close()`。
Ray 验证仍待运行完成，不沿用 standalone 的 PASS。

## 接口

仿真器在 AppLauncher 之后导入：

```python
from pathlib import Path
from isaaclab.app import AppLauncher

launcher = AppLauncher(
    headless=True, enable_cameras=True, kit_args="--/renderer/multiGpu/enabled=false"
)

import torch
from carrot_sim.arena_libero.config import ArenaLiberoConfig
from carrot_sim.arena_libero.environment import make_env

env = make_env(ArenaLiberoConfig(asset_root=Path("/root/arena-libero-assets")))
try:
    obs = env.reset(seed=42)
    action = torch.zeros(env.num_envs, 7, device=env.device)
    action[:, 6] = -1  # 张开夹爪
    obs, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
    launcher.app.close()
```

观测为平铺 dict：`state[N,8]`、`critic[N,52]`、`image[N,H,W,3]`、
`wrist_image[N,H,W,3]`。状态 float32，图像 uint8；全部为自有存储的 GPU tensor。
动作 float32 `[N,7]`，范围 [-1,1]；前 3 维缩放为每步 0.02m 的机器人基坐标系位移，
后 3 个旋转分量每步 0.1rad，末维负数打开、非负关闭。

`env.reset(env_ids=...)` 接收本 GPU 上的一维 int64 子集，返回全量观测。
`step()` 同步自动 reset；`info["final_observation"]` 只在 `final_mask` 为真处有效。
critic 在纯超时处使用 final observation bootstrap，在 terminated 处不 bootstrap。
`action_space`/`observation_space` 提供 Gymnasium shape/dtype 描述，数据通路保持 GPU tensor。


## Ray 验证进度

2026-09-22：检测到已有 Ray 2.58.0 单节点集群，224 CPU / 8 GPU 均无占用；
复用既有 head，测试只创建、释放自己的 placement group 和 actor。没有启动第二 head。
准备运行 `20260922-ray-01`；GPU 前暂停 dguard 并检查无 renderer，
测试退出后保留原 Ray head 并恢复 dguard。

ray-01（`5f3f1fce-0666`）：连接成功，但测试错误地要求 `CUDA_VISIBLE_DEVICES == "0"`，
在 Kit 启动前退出 1。Carrot runtime 有意保留可见性，通过 `Worker.local_rank` 映射分配设备；
已将验收改为从 local_rank 和现有可见性计算物理 GPU，并要求实际为 GPU 0。
没有改动 distributed runtime，也没有扩大资源预留或重启既有 head。继续 ray-02。

## 2026-09-22 ray-02

任务 ID：`5f3f1fce-0672`，输出 `${MY_DFS}/benchmarks/arena-libero-rl/20260922-ray-02`。
真实 Ray actor 内 PPO / success / reset / terminal 验收通过，driver exit 0；
worker PID 1653341，Python `/opt/venvs/carrot/bin/python` 3.12.13，Ray 2.58.0，
local_rank=0 / physical_gpu=0。再次 RPC identity 成功；actor 与 placement 已释放，原 head 保留。
独立 CPU 测试 11/11、产物 validator 通过；无新 Xid，dguard 已恢复。

视觉时序验收未通过：主模型查看 reset_wrist_image.png 发现首帧仍拍到旧机械臂姿态，
与 5 个中性动作步之后的 wrist_image.png 平均绝对像素差约 52.1（0–255）。
已把初始物理/渲染预热放入 make_env，随后按原 seed 完整 reset；
运行期的部分 reset 不执行额外物理步。新测试将检查首帧差异和移动手腕后的即时 reset 图像。


## 2026-09-22 ray-03

任务 ID：`5f3f1fce-0680`，输出 `${MY_DFS}/benchmarks/arena-libero-rl/20260922-ray-03`。
初始化预热后，首帧与预热图的 MAE 降至 3.73；重复 reset 图 MAE=2.47。
主模型已独立查看首帧、移动后、再次 reset 的腕部图，均能看到碗与盘子。
Ray/PPO/成功/reset/terminal 均 PASS、driver exit 0，CPU 11 项和独立 validator 通过。

检查移动后图像时发现变化较小，追加物理控制验收：每个环境发送 12 个向上动作步，
记录控制前后 TCP 与 joint positions/targets，要求 TCP 实际移动超过 1cm；
再发送 8 步夹爪关闭，要求指关节间隙实际变化超过 1cm。
该断言用于防止“优化器更新了，但模拟器未响应动作”的假阳性，等待 ray-04。


## 2026-09-22 ray-04：最终验收 PASS

任务 ID：`5f3f1fce-0689`，driver exit 0。完整产物目录：
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/arena-libero-rl/20260922-ray-04/`。
主模型独立读取 `result.json`、`ray_result.json`、`control.json` 并检查腕部首帧、
移动后和再次 reset 的图片；独立产物 validator 通过，CPU pytest 11 passed。

| 检查 | 结果 |
| --- | --- |
| 运行时 | Python 3.12.13 / Torch 2.11.0+cu128 / Sim 6.0.1.0 / Lab 3.0.0b2.post1 / Arena 0.3.0 / Ray 2.58.0 |
| Ray actor | PID 1658078，`/opt/venvs/carrot/bin/python`；local_rank=0，physical_gpu=0；运行后再次 RPC 成功 |
| 并行仿真 | 1 个 H20 renderer，4 environments，两个 RGB 相机 |
| 实际动作响应 | 4/4 环境 TCP 位移至少 16.413 mm；指关节平均开合变化至少 32.313 mm |
| PPO | 128 transitions，8 truncations，2 次参数更新；actor 最大更新 0.0006003864，critic 0.0005998611 |
| RL 语义 | 接触成功 fixture、部分 reset 隔离、terminal observation、超时 bootstrap 均 PASS |
| reset 图像 | 首帧 MAE=3.732，移动后再 reset MAE=2.616（0–255）；均低于 20 门槛 |
| LW 代码依赖 | `lw_imported=false`，只复用 USD 资产 |

执行代理收尾核查：自有 actor / placement 已释放，既有 Ray head 保留，
资源回到 0/224 CPU、0/8 GPU；无残留测试 renderer，dmesg 未见 Xid。
dguard 已恢复，恢复定时器已取消。本轮没有安装或升级依赖。

验收范围是黑碗放盘子单任务及真实 RL 训练链路；成功 fixture 通过物理放置验证判据，
PPO 短跑验证数据和参数更新，不代表策略已经学会任务，也不代表完整 LIBERO benchmark 等价性。
