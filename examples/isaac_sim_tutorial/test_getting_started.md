# Isaac Sim Getting Started

状态：PASS（2026-10-26，真实 GPU / PhysX 三轮验证）。

当前入口直接用 shell 启动官方教程 Python，运行完三轮后正常退出即完成运行检查。
不使用 pytest、AST 改写或采样包装；无桌面模式、GPU 和线程数通过 Kit 命令行参数指定。
教程保留三轮逻辑，新增可选 --record 参数；环境仅传必需路径，避免把会话凭据打印到日志。

运行前通过当前 Gemini 会话确认 MY_DFS，选空闲 GPU，暂停 dguard，并核对文件已同步。
设置 ISAAC_SIM_GPU 后执行：

```bash
bash examples/isaac_sim_tutorial/test_getting_started.sh
```

日志直接输出到终端，额外 Kit 参数可以追加到 shell 命令后。
下面保留之前使用采样包装得到的物理验证结果，相关 Python 包装文件已按用户要求删除。

环境预检：Isaac Sim 6.0.1.0，Python /opt/venvs/carrot；GPU 为 RTX PRO 5000 Blackwell。
Carrot 基线 commit：fcd7a972995d6ac6e13176de8a9785e76b180b5e；本次测试代码尚未提交。
上游源码 tag：v5.1.0，commit 47d886f2858d1ceed556b21c88927aa67bc81c12。
Docker image tag、容器内 Isaac Sim 构建 commit：未记录。
教程 SHA256：ad22a223f704cc2df1f12fbf3c63f0d0e8567f1b739e42b3a1dc377f345903dd。

## 2026-10-26 历史物理验证（已删除采样包装）

最终任务 `079aad86-0497` 退出 0，pytest **1 passed in 28.85s**。
Python 3.12.13、Isaac Sim 6.0.1.0、NumPy 2.3.1、pytest 9.1.1；
GPU 1 为 NVIDIA RTX PRO 5000 72GB Blackwell，UUID 前缀 `8a5b4e94`。
测试在 headless 模式下实际渲染并推进 PhysX；没有进行 GUI 窗口交互验收。

三轮各 100 帧，位置数据形状为 (300, 3, 3)，全部有限。三列对象依次为黄色、绿色、青色。

| 轮次 | 黄色方块终点 z（米） | 绿色方块终点 z（米） | 青色方块终点 z（米） |
| --- | --- | --- | --- |
| 0：黄色只有视觉属性 | 1.000000 | 0.500000 | 0.150004 |
| 1：黄色添加刚体属性 | -12.761254 | 0.500000 | 0.150004 |
| 2：黄色添加碰撞属性 | 0.150001 | 0.500000 | 0.150004 |

黄色 schema 也按预期累积：第一轮无刚体、碰撞 API；第二轮有 PhysicsRigidBodyAPI；
第三轮同时有 PhysicsRigidBodyAPI 和 PhysicsCollisionAPI。
第三轮黄色最低中心高度 0.114375 米，存在瞬态接触穿透，最终回到约 0.15 米；
本测试容差为接触中心高度 (0.10, 0.20) 米，验证能承接方块，不作为接触精度基准。
青色各轮最低中心高度均约 0.150000 米，绿色全部坐标始终不变。

当前会话只发现一个可用团队目录，实际挂载类型为平台的 fuse.dop-fuse。
MY_DFS 为 `/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu`；经 Mutagen carrot-zjk1 同步并核对 SHA256。
当时使用旧版 shell 和采样包装的实际执行参数（作为历史记录保留）：

```bash
export MY_DFS=/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu
export ISAAC_SIM_GPU=1
export ISAAC_SIM_RUN_DIR="$MY_DFS/tests/carrot/isaac_sim_tutorial/getting-started-20261026-02"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
bash examples/isaac_sim_tutorial/test_getting_started.sh
```

此证据目录已存在，重跑必须更换 ISAAC_SIM_RUN_DIR 的最后一级目录。
测试数据和日志在上述目录的 `test_getting_started_physics0/` 下，pytest.log 在其父目录。
实际执行的测试 Python SHA256：
`db59bb6e038c7e90926b75c56702e54f803157fd065f8dfd9a347ef91f89a949`。

本地 Python 语法、bash -n 通过；远端测试文件 ruff check、ruff format --check 通过。
教程保留上游未使用的循环变量 j，ruff 检查时显式排除 B007；其余规则通过。

首次运行也通过（27.54 秒），但 Kit 辅助进程的错误日志包含了继承环境中的凭据。
已对首次生成日志脱敏并释放对应 MCP 任务输出；随后将测试子进程改为仅继承明确列出的
路径、语言和 GPU 环境变量，完成上述第二轮验证。最终生成日志的凭据值扫描通过。
Kit 仍有音频设备和辅助进程告警，进程正常退出，物理断言全部通过。

测试进程已结束，GPU 1 显存恢复到测试前水平；原 GPU 0 用户进程 PID 630356 仍在。
dguard 已恢复 DGUARD_WATCH=1，无待执行的延迟恢复任务。

## 2026-10-26 直接运行与录像

状态：PASS，shell 直接运行与录制均已通过。
已删除 AST/pytest 包装，shell 直接调用 getting_started.py。
追加 `--record /absolute/path/to/getting_started.mp4` 录制当前渲染视口。
每个物理步保存一帧，共 300 帧；默认 60 Hz 物理，以 30 FPS 半速播放，总长 10 秒。
画面标注轮次和步数，不包含桌面菜单；只调整录制视角，不改变方块的物理逻辑。

最终任务 `079aad86-0509` 退出 0，耗时约 27 秒，三轮日志完整，输出 Recorded 300 frames。
实际运行命令（MY_DFS 来自本次已确认会话）：

```bash
export MY_DFS=/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu
export ISAAC_SIM_GPU=1
RUN_DIR="$MY_DFS/tests/carrot/isaac_sim_tutorial/getting-started-recording-20261026-03"
bash examples/isaac_sim_tutorial/test_getting_started.sh \
  --record "$RUN_DIR/getting_started.mp4" "--/log/file=$RUN_DIR/kit.log"
```

远端证据目录为上述 RUN_DIR，保存 getting_started.mp4、kit.log、run.log。
已取回本地 `~/Downloads/isaac_sim_tutorial/getting_started-20261026.mp4`；
三轮抽帧预览为同目录 `getting_started-20261026-preview.png`。
ffprobe 独立验证：H.264、1280×720、30 FPS、300 帧、10 秒、194028 字节；
ffmpeg 全片解码通过，目视抽查三轮画面、标签和对象均完整可见。
前两版录像均正常退出，但分别存在遮挡和起始方块靠近画面边缘，已调整视角后重录。
最终源码 SHA256：`df7a406c901ad6a966620a7fbdfc5951153b804994dd761096a528f61d39ce6a`；
shell SHA256：`e04fb3ece4acdcc4b6060d8c2efa42cb41eb740082cb541e4f89cdba93ee06d0`。
Python 语法、bash -n、ruff check 和 ruff format --check 均通过。
环境仍为 Isaac Sim 6.0.1.0 / Python 3.12.13 / GPU 1 RTX PRO 5000；Docker tag 未记录。
录制后 dguard 已恢复 DGUARD_WATCH=1，无待执行的延迟恢复任务。
