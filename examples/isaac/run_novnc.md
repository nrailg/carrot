# 在 headless GPU server 上使用 noVNC

MacBook 只需要浏览器。服务器使用 Xvfb 创建虚拟显示器，Openbox 管理窗口，
x11vnc 传送桌面，websockify/noVNC 提供网页和 WebSocket 连接。
Isaac 使用 GPU 渲染，VNC 只负责传输桌面画面。

## 安装（Ubuntu 22.04）

在 GPU server 执行。需要 root 权限；软件包安装在容器内，换容器后需重新安装。

```bash
# 当前 Gemini 环境访问外网使用已配置的代理。
source /tmp/set_proxy.sh
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    xvfb x11vnc openbox novnc websockify x11-utils x11-xserver-utils xterm
```

Python 环境和 Isaac 资源沿用 [spawn_prims_assets.md](spawn_prims_assets.md)。
`DFS` 为当前服务器实际挂载的个人持久化目录，不是团队目录。

## 启动

先检查端口和 GPU 看护程序：

```bash
ss -lntup | grep -E ':808[01]\b|:6080\b|:5900\b'
bash /root/dguard/dguard.sh status --local
# 按实际使用时长暂停；这里以两小时为例。
bash /root/dguard/dguard.sh stop 120
```

已有 Jupyter 占用 8080/8081 时，不要直接覆盖或停止它。先选择空闲端口，或在确认
可以停止对应 Jupyter 后释放端口。`run_novnc.sh` 默认使用 TCP 8080。

```bash
export DFS='<当前个人 DFS 目录>'
cd "$DFS/work/carrot"
bash examples/isaac/run_novnc.sh
```

脚本启动 Xvfb `:99`（1280×800）、Openbox、仅本机可访问的 VNC TCP 5900、
noVNC 和 `spawn_prims.py --viz kit`。不要加 `--headless` 或 `--livestream`。
只在 GPU 0 渲染，使用 CPU 0–15。可用 `DISPLAY`、`VNC_PORT`、`NOVNC_PORT`、
`NOVNC_BIND` 和 `ISAAC_CPUSET` 显式调整配置。

### 换一个 Python 程序

停止当前启动脚本后，把 Python 文件路径作为第一个参数，其余参数原样传给程序：

```bash
cd "$DFS/work/carrot"
bash examples/isaac/run_novnc.sh examples/isaac/spawn_prims.py --num_steps 1000
# 运行两个 Cartpole：随机施加关节力，每 500 步重置状态。
bash examples/isaac/run_novnc.sh examples/isaac/run_articulation.py
# 四个软方块：两个自由落体，两个通过一个顶点拉伸，每三秒重置。
bash examples/isaac/run_novnc.sh examples/isaac/run_deformable_object.py
# 两个 Surface Gripper，随机发送吸盘开闭命令；此版本要求 CPU 物理。
bash examples/isaac/run_novnc.sh examples/isaac/run_surface_gripper.py --device cpu
# 换成你的程序；相对路径以当前工作目录为准，也可以传绝对路径。
bash examples/isaac/run_novnc.sh /path/to/your_isaac_program.py
```

不传文件路径时仍运行 `spawn_prims.py`。目标程序需要支持 Isaac Lab `AppLauncher`
的 `--viz kit` 和 `--kit_args` 参数；普通 Python 程序不能直接套用这些参数。
程序退出后，脚本会一并停止桌面服务。

当前受控环境使用 `x11vnc -nopw`，无需密码，也不生成密码文件。

首次窗口模式启动可能需要编译 RTX shader。Kit 日志中的
`Waiting for RtPso async group async compilation` 表示仍在等待编译，不能只凭网页
能打开就认定 Isaac 场景已就绪。要同时检查终端出现 `Setup complete...` 和浏览器内
确实显示场景。

## MacBook 访问

通过平台提供的 **对应端口访问链接** 打开 `vnc.html`，点击 Connect 即可进入桌面。
如果允许直连，可以使用 `http://<GPU IP>:8080/vnc.html`。
平台转发需要支持 WebSocket；本方案不使用 UDP。

办公网络可能对未开放的端口返回 IT 提醒、502 或 `acl_denied`，此时
通过 DevCloud 的普通 SSH 转发也不一定可用。应使用平台已授权的端口访问入口，
不要把 `nc` 的 TCP 连接成功当作 HTTP/VNC 服务可访问的证据。

进入桌面后可以操作 Isaac 的窗口和视口。noVNC 左侧工具栏可以调整缩放、发送特殊
按键和断开连接。断开浏览器不会停止服务器仿真。

## 停止

在启动脚本的终端按 Ctrl+C。脚本只停止自己启动的 Isaac、websockify、x11vnc、
Openbox 和 Xvfb，不停止已有 Jupyter 或其他 Python 任务。通过 Gemini MCP 后台
启动时，停止对应后台任务；不要执行 `pkill python`。

## 本次部署记录（2026-10-04）

- Ubuntu 22.04.5 LTS，NVIDIA RTX PRO 5000 Blackwell。
- 安装成功：Xvfb、Openbox、x11vnc、noVNC、websockify。
- noVNC 在服务器 TCP 6080 启动成功，本机请求 `/vnc.html` 返回 HTTP 200。
- 用户确认后停止原 TCP 8080/8081 的两个 Jupyter Lab，noVNC 改用 8080。
- MacBook 直连 6080 返回办公网络错误；DevCloud 转发返回 `403 acl_denied`。
- MacBook Chrome 访问 `http://100.77.131.77:8080/vnc.html` 成功，WebSocket 和 VNC
  连接成功，桌面显示 Isaac Lab 窗口和场景。按用户要求改为免密码连接。
- 在浏览器关闭浮动面板、点击暂停和恢复按钮成功，确认鼠标输入可控制 Isaac。
- 首次 shader 编译约四分钟；终端出现 `Setup complete...`，仿真步数持续增长。
- 容器将 `localhost` 的 IPv4 解析为服务器内网 IP，因此 websockify 必须使用
  `127.0.0.1:5900`；x11vnc 使用 `-threads -noxrecord -noscr` 避免连接握手迟滞。
- 部署日志位于 `$DFS/work/carrot/outputs/isaac/novnc/install.log` 和 `first_start.log`。

参考：[noVNC 官方项目](https://github.com/novnc/noVNC)。

## 完整 Isaac Sim GUI 与 ROS2 Bridge（2026-10-27）

`run_novnc.sh` 的目标是 Isaac Lab Python 示例。完整编辑器的入口为
`isaacsim isaacsim.exp.full`，需要先启动上述 Xvfb、Openbox、x11vnc 和 websockify。
在已包含桌面组件的镜像中先检查命令是否存在，不要重复安装软件。
本次采用 `x11vnc -nopw` 免密码连接，VNC 仍仅监听 localhost，websockify 绑定指定的服务器 IP。

Ubuntu 22.04、Isaac Sim 6.0.1.0、Python 3.12 环境使用内置 ROS2 Humble。
在启动 Isaac 的同一个 shell 中设置以下变量（仅影响此进程及子进程）：

```bash
source /opt/venvs/carrot/bin/activate
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_LIB_DIR=/opt/venvs/carrot/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib
test -d "$ROS_LIB_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROS_LIB_DIR"

# DISPLAY、DFS 和桌面服务需已准备好；代理沿用当前环境已授权配置。
taskset -c 0-15 isaacsim isaacsim.exp.full --allow-root \
    --/renderer/activeGpu=0 --/physics/cudaDevice=0 \
    --/renderer/multiGpu/enabled=false \
    --/app/window/width=1600 --/app/window/height=900 \
    --/plugins/carb.tasking.plugin/threadCount=16 \
    --/plugins/omni.tbb.globalcontrol/maxThreadCount=16 \
    "--/persistent/isaac/asset_root/default=$DFS/isaacsim_assets/Assets/Isaac/6.0"
```

如果外层使用 `env -i` 清理环境，必须把 `ROS_DISTRO`、`RMW_IMPLEMENTATION` 和
补齐后的 `LD_LIBRARY_PATH` 显式传入；只在外层 `export` 会被清掉。
发行版和路径应匹配实际安装版本，不能将这里的 Humble 路径直接套到其他版本。

本次故障是启动环境遗漏：日志报 `AMENT_PREFIX_PATH is not set or empty`、
`RMW was not loaded` 和 `ROS2 Bridge startup failed`。内置库文件存在，但默认搜索路径
找不到其依赖。使用内置库时，本次无需安装系统 ROS2，也无需手工编造 `AMENT_PREFIX_PATH`。

补齐变量后，最终日志出现 `rcl::init()`、`rclpy loaded`、
`ROS bridge extension isaacsim.ros2.bridge enabled successfully`，约 20 秒后 `app ready`；
进程映射确认加载了内置 `librcl.so`、`librmw_implementation.so` 和
`librmw_fastrtps_cpp.so`。浏览器免密码连接成功，GUI 视口约 114 FPS。
验证范围是 Bridge 初始化和 GUI 显示，尚未验证 ROS2 topic 收发。

第一次重启把 ROS 库目录放在搜索路径前端，Bridge 初始化后进程以 139 退出；
改为按官方示例追加在末尾后重启成功。未做重复对照实验，不能认定路径顺序就是崩溃根因。
全过程未安装或升级依赖、未修改全局 shell 配置；telemetry 子进程 fork 超时仍可能出现。

参考：[NVIDIA 内置 ROS2 库配置](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_ros_other_platforms.html)。

## 可变形物体示例

[run_deformable_object.py](run_deformable_object.py) 对应官方
[Deformable Object 教程](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/run_deformable_object.html)。
保留四个软方块的节点状态、随机姿态重置和运动学顶点约束逻辑；代码基于官方
[v3.0.0-beta2 示例](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/scripts/tutorials/01_assets/run_deformable_object.py)，
使用服务器 Isaac Lab `3.0.0b2.post1` 的 `AppLauncher` 和 PhysX 接口。

需要已安装 `pytetwild`，用于把方块表面网格转换为四面体体积网格。
地面使用 `$DFS/isaacsim_assets/Assets/Isaac/6.0` 中的已有资产，无需另外下载软体模型。
浏览器中的粉色球是受约束顶点的目标位置；第 0、3 个方块被缓慢向上提拉，
其余两个受重力落到地面。仿真运行时每三秒随机重置一次，可用工具栏暂停观察。

2026-10-08 在 RTX PRO 5000 Blackwell、Isaac Sim `6.0.1.0`、Isaac Lab
`3.0.0b2.post1` 上验证：`Setup complete...` 后仿真时间持续增长，完成多轮重置；
MacBook Chrome 通过 TCP 8080 连接，看到四个软方块、两个顶点目标标记和落地/提拉运动。
Kit 仍会报告 telemetry 子进程 fork 失败；本次场景在该报错之后持续正常运行。

## Surface Gripper 示例

[run_surface_gripper.py](run_surface_gripper.py) 对应官方
[Surface Gripper 教程](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/run_surface_gripper.html)，
基于匹配服务器接口的
[v3.0.0-beta2 示例](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/scripts/tutorials/01_assets/run_surface_gripper.py)。
创建两个 PickAndPlace 机器人及其 Surface Gripper，每 500 步重置，逐步发送随机开闭命令。
相机同时覆盖两个机器人，每 50 步打印命令和状态。

服务器 Isaac Lab `3.0.0b2.post1` 的 Surface Gripper 要求 CPU 物理，因此示例默认
`--device cpu`，显式指定其他设备会报错；Kit 仍使用 GPU 渲染。
机器人资产使用已有完整包中的
`$DFS/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Tests/PickAndPlace/pick_and_place_robot.usd`。
使用此版本的 `set_grippers_command_index` 写入命令，不需要新增依赖或下载资产。

命令小于 -0.3 时打开，大于 0.3 时尝试吸附，中间区间不发起新的开闭操作。
返回状态 -1 为 Open、0 为 Closing、1 为 Closed；发出 Closing 命令不代表吸附成功，
需要附近存在符合条件的碰撞刚体。本例沿用官方随机命令，不包含完整的物体搬运轨迹。

2026-10-08 在 RTX PRO 5000 Blackwell、Isaac Sim `6.0.1.0`、Isaac Lab
`3.0.0b2.post1` 上验证：最终代码运行超过 500 步并完成机器人和吸盘重置，
状态返回 Open/Closing；MacBook Chrome 经 TCP 8080 连接成功，显示两套机器人。
本次未验证到 Closed 状态或物体搬运成功。telemetry 子进程仍有 fork 失败报错，
主仿真持续运行。首次启动日志保存在
`$DFS/work/carrot/outputs/isaac/novnc/surface_gripper_first_start.log`。
