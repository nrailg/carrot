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
