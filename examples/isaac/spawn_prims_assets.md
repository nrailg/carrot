# Isaac Sim 完整资源包

[download_isaacsim_assets.sh](download_isaacsim_assets.sh) 已改为下载官方完整资源包。
当前服务器使用 Isaac Sim 6.0.1，脚本固定使用官方 6.0.0 资源包，解压后的资源版本目录是
`Assets/Isaac/6.0`。用户提供的 5.1 文档使用相同流程，但下载地址、MD5 和目录版本不同；
本脚本采用 [6.0 官方说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack)。

## 下载、校验、合并、解压

先将 `DFS` 设置为当前用户的个人持久化目录（不是团队挂载根目录），
再在 GPU server 的 Carrot 根目录执行。本次已确认的路径为：

```bash
export DFS=/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu
bash examples/isaac/download_isaacsim_assets.sh
```

脚本不接受位置参数，也不猜测 DFS。目录均从 `$DFS` 派生：

- 下载目录：`$DFS/isaacsim_assets/downloads/6.0.0/`
- 解压目录：`$DFS/isaacsim_assets/`
- 资源根目录：`$DFS/isaacsim_assets/Assets/Isaac/6.0/`
- 环境文件：`$DFS/isaacsim_assets/asset_env.sh`

使用上述 `DFS` 时，与之前脚本的默认目录一致，已有文件无需移动或重新下载。

脚本依次完成：

1. 写入并加载 `/tmp/set_proxy.sh`，按 `wx-download` 设置代理和 `no_proxy`。
2. 从 `https://downloads.isaacsim.nvidia.com` 下载五个分片，支持断点续传。
3. 按官方 MD5 校验所有分片；已下载且校验通过的分片直接复用。
4. 按 `.001.zip` 至 `.005.zip` 顺序合并为一个 ZIP。
5. 用 `unzip -o` 解压到目标目录，覆盖同名资产并通过解压时的 CRC 校验。
6. 检查资源根目录包含 `Isaac`、`NVIDIA`，以及教程的地面和桌子 USD。
7. 生成 `asset_env.sh`，供仿真示例加载本地资源路径。

依赖为 Linux 上的 `bash`、`curl`、`md5sum`、`unzip` 及常用系统命令；脚本不安装依赖。
校验失败立即退出，提示需要移走的具体文件，不自动删除已有下载。
完整包体积较大，磁盘需同时容纳五个分片、合并后的 ZIP 和解压资源。
分片和合并 ZIP 保留在 `isaacsim_assets/downloads/6.0.0/`，确认解压成功后可自行清理。
无需 USD / URDF 转换，保留官方目录结构即可。

## 可变形体依赖

教程的 `MeshCuboidCfg` 需要 `pytetwild` 自动生成体积四面体网格。
当前 Isaac Lab 的 `all` extra 没有包含它；在容器依赖环境补装：

```bash
source /opt/venvs/carrot/bin/activate
source /tmp/set_proxy.sh
uv pip install --python /opt/venvs/carrot/bin/python --index-url https://pypi.org/simple \
  'pytetwild[all]==0.4.2'
```

`[all]` 包含该版本导入所需的 `pyvista`。这是补装依赖，不将 Carrot 安装进 venv。
示例项目和主项目的 Isaac 依赖配置也记录了这个版本。

## 运行 spawn_prims.py

示例已接入现有的 `asset_root.py`，在导入 `isaaclab.sim` 前覆盖 Isaac Lab 资源常量。
完整包准备好后：

```bash
source /opt/venvs/carrot/bin/activate
source "${DFS:?请先设置个人 DFS}/isaacsim_assets/asset_env.sh"
unset CUDA_VISIBLE_DEVICES
python examples/isaac/spawn_prims.py --headless --device cuda:0 --num_steps 300 \
  --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
```

从 Carrot 根目录运行。`--num_steps 300` 运行 300 步后关闭应用；省略该参数或设为 0 时持续运行，用 Ctrl+C 停止。
每 100 步打印进度。脚本排除 `omni.kit.telemetry` 扩展，避免当前容器启动遥测子进程失败。
通过 `--device` 和 Kit 的 `activeGpu` 选择设备，helper 不再自动设置 `CUDA_VISIBLE_DEVICES`。
运行时使用与下载时相同的 `DFS`。
启动前按远程运行规范检查并暂停 dguard。

## Headless 录制 MP4

`--record` 自动启用相机渲染，输出 640×480、25 FPS 的 H.264 MP4，以及同名首帧 PNG。
物理步长是 0.01 秒，每四步保存一帧；300 步对应 75 帧、3 秒视频。
录像时必须显式指定至少 4 步，防止持续仿真无限写入视频。

当前 Gemini 容器有两份 NVIDIA Vulkan ICD 清单，指定 `/etc` 中这一份可避免 GPU 重复枚举。
机器有 768 个逻辑 CPU，示例命令限制到 16 核，避免启动大量渲染线程：

```bash
source /opt/venvs/carrot/bin/activate
source "${DFS:?请先设置个人 DFS}/isaacsim_assets/asset_env.sh"
unset CUDA_VISIBLE_DEVICES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
taskset -c 0-15 python examples/isaac/spawn_prims.py \
  --headless --device cuda:0 --num_steps 300 \
  --record "$DFS/work/carrot/outputs/isaac/spawn_prims.mp4" \
  --kit_args '--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=16'
```

录像依赖 `imageio[ffmpeg]==2.37.2`，当前容器已经具备，两个项目依赖配置均已补充。
这些 ICD 路径和 CPU 编号是当前容器实测值；更换服务器时应核对。
首次 RTX 渲染需要编译着色器，启动会比纯物理仿真慢。

2026-10-04 实测 PASS：退出码 0，MP4 为 640×480、25 FPS、75 帧、3 秒，
成功解码并检查首末帧变化；首帧可见红色圆锥、绿色刚体圆锥、蓝色可变形体及桌子。
Vulkan 初始化仍报告一次 `nvidia-smi` 辅助进程启动失败，但本次视频生成成功。
远端输出：`$DFS/work/carrot/outputs/isaac/spawn_prims.mp4`、同名 `.png`。

## 验证状态

已核对官方下载地址、五个 MD5、合并/解压方式；Bash 语法和 Python 语法检查通过。
2026-10-04：在指定 launcher、RTX PRO 5000 上完成 300 步（dt=0.01，3 秒仿真时间），
退出码 0；场景包含教程的刚体、可变形体和本地桌子 USD。遥测启动错误及
`CUDA_VISIBLE_DEVICES` 警告未再出现。AppLauncher 启动后重新应用资源配置，
并显式设置地面 USD 路径，避免云端路径查询阻塞。

远端日志：`$DFS/work/carrot/outputs/isaac_spawn_prims_20261004_pass.log`。
仍有上游 protobuf 重复注册和可选扩展警告，但本次仿真正常完成。
这次验证证明示例能初始化并完成循环，不包含画面或物理数值精度校验。
