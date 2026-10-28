# Isaac 机器人工作台：Linux 编程与临摹

一个 example，两种任务，使用 Franka Panda 与相机观测。
`computer` 使用两台机械臂、两个刚性笔尖和 18 维动作，通过真实接触按下弹簧键帽，向独立的 QEMU Linux guest 发送 HID；
`paint` 通过笔尖接触在画布留下笔迹，临摹显示器上的参考图。
绘画保留单臂、9 维动作。每条臂均为 7 个臂关节加 2 个夹爪关节，专家将夹爪保持闭合。
支持一个 Isaac World 中的多个环境，以及按环境编号局部 reset。

这是可供 RL / 模仿学习接入的环境与示范控制器。`scripted` 使用特权几何信息做 IK，
不是训练得到的视觉策略；当前没有宣称机器人已经学会编程、任意绘画或跨任务迁移。
画笔为刚性笔尖与接触驱动的单色栅格笔迹，不模拟柔性刷毛、颜料混合和流体。

## 运行

已验证 Linux x86_64、Isaac Sim 6.0.1.0、Isaac Lab 3.0.0b2.post1、QEMU 6.2 + KVM。
Python 环境使用容器预装的 `/opt/venvs/carrot`。额外系统依赖为
`qemu-system-x86`、`busybox-static` 和可启动的 Linux kernel。
没有 KVM 时可显式选 `--accelerator tcg`，本次未验证 TCG，不自动降级。

先确认自己的 DFS 路径，将 `MY_DFS` 设为该目录；不要从其他机器复制 mount 名称。

```bash
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ROBOT_USD="${MY_DFS}/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
RUN_DIR="${MY_DFS}/benchmarks/isaac-workbench/my-run"

# 两个环境临摹同一幅线稿。
python examples/isaac_workbench/run.py --task paint --num-envs 2 \
  --robot-usd "$ROBOT_USD" --output "${RUN_DIR}/paint" --gpu 1 --headless \
  --check --partial-reset-check

# 四个环境分别验证大小写、求和、最大值与字符串反转。
python examples/isaac_workbench/run.py --task computer --num-envs 4 \
  --exercise shift_p,sum_two,max_list,reverse_text --guest-dir "$WORKBENCH_GUEST" \
  --robot-usd "$ROBOT_USD" --output "${RUN_DIR}/computer" --gpu 1 --headless \
  --check --partial-reset-check
```

每次使用新的输出目录。大型 CPU 主机可在 `python` 前加 `taskset -c 32-47` 限制线程亲和性；
具体 CPU 范围应在当前节点确认。本次节点存在重复 Vulkan ICD 配置，显式使用
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`，不要无条件照搬到其他主机。
不设置 `CUDA_VISIBLE_DEVICES`，由 `--gpu` 同时指定渲染与物理设备。

准备 guest（一次即可）：从本机发行版取得 kernel 包并解压到 `KERNEL_ROOT`，
其 `boot/` 目录需要同时有 `vmlinuz-${KERNEL_VERSION}` 和 `config-${KERNEL_VERSION}`。
构建器检查键盘、VGA、devtmpfs 驱动为内建，复制系统 Python 与动态依赖，生成只读基础 initramfs。
已验证 Ubuntu 22.04 的 `5.15.0-198-generic`、`/usr/bin/python3.10`、`/usr/lib/python3.10`。

```bash
python examples/isaac_workbench/prepare_guest.py \
  --kernel "${KERNEL_ROOT}/boot/vmlinuz-${KERNEL_VERSION}" \
  --kernel-root "$KERNEL_ROOT" --kernel-version "$KERNEL_VERSION" \
  --output "$WORKBENCH_GUEST"
```

输出 `vmlinuz`、`initramfs.cpio.gz`、含 SHA256 的 `manifest.json`。基础镜像不含账号凭据、
宿主目录共享或网络。每次 episode 另加只属于该 VM 的题目 initramfs overlay。
修改 `guest_task.py` / `guest_console.py` 后需重新构建基础 guest。

## 任务

| ID | 屏幕上的任务 | 示范表达式 | 判定 |
| --- | --- | --- | --- |
| `shift_p` | 返回大小写严格匹配的字符串 Pp | `'Pp'` | Shift+P 后释放 Shift，再按 p |
| `sum_two` | 返回整数 a 与 b 的和 | `a+b` | 正数、负数、零，3 个用例 |
| `max_list` | 返回非空列表 values 的最大值 | `max(values)` | 正数、全负数、单元素，3 个用例 |
| `reverse_text` | 反转字符串 text，保留空格 | `text[::-1]` | 普通字符串、空格、空串，3 个用例 |

这是 Python **表达式编程**题，不是完整 IDE 或任意多文件项目评测。
题目显示在 guest 的 Linux console，Enter 提交，Backspace 修改。
评分在真实 guest 内执行，测试用例不进入 policy observation；串口只回传可信评分。
AST 白名单限制 IO、属性访问与 import，单用例有 1 秒 CPU 执行超时，guest Python 有内存上限。
`tasks.py` 是增加题目的入口；示范程序必须能由当前键盘输入。
键帽只对应单个 HID code，没有符号宏键或粘滞修饰键。
输入大写或上档符号时，第二臂实际压住 Shift，第一臂按相应字符键，随后分别抬起。
例如 `(` 是两个笔尖实际按住 Shift 和 9。`typing_chord` 只属于专家打字动作规划，
VM 桥接逐条转发物理边沿，不补发修饰键。RL 策略直接输出两臂关节动作，不经过字符规划器。
Shift / Ctrl 键放在键盘左侧独立位置，键距适配笔尖；布局不是完整商用键盘复刻。

`paint` 默认参考图是房屋线稿，包含多段轨迹和抬笔。`--strokes expert.json`
可提供另一组示范笔画（JSON 格式为 `[[[u,v], ...], ...]`，坐标在 `[0,1]`）；
默认目标图由这些笔画栅格化。`--reference-image target.png` 可改为外部参考图片。
外部图片搭配 `scripted` 必须同时提供对应的 `--strokes`，没有自动读取目标图生成解答的逻辑。
学习策略只需目标像素，不需要专家笔画。当前 IoU 指标比较前景线条，适合单色白底临摹，
不评估颜色、风格或照片的语义相似度。

## RL 接口与 QEMU 并行约束

先创建 `SimulationApp`，再导入 `VectorWorkbench`。`run.py` 展示了完整生命周期。

| 接口 | 契约 |
| --- | --- |
| `step(actions)` | `float[N,D]` 绝对关节位置，computer D=18、paint D=9；每 9 维内前 7 维 rad、末 2 维 m；有限且在关节限位内 |
| 观测 | `rgb[N,H,W,3]`、`joint_positions[N,D]`、`joint_velocities[N,D]`；paint 另有 `reference_rgb[N,H,W,3]` |
| 返回 | `(obs, reward[N], terminated[N], truncated[N], info)`；数组为 NumPy |
| 物理时钟 | 240 Hz；每个控制 step 推进 4 个物理 tick；每 tick 读取接触力 |
| 奖励 | computer：通过用例比例的增量；paint：前景 IoU 增量 |
| 成功 | computer 全用例通过且键已释放；paint IoU ≥ 0.72 且抬笔、无接触 |
| `reset(env_ids)` | 只重置指定环境，返回观测与 reset 编号；不推进其他环境的物理或 guest |
| episode 结束 | **不自动 reset**；先使用终止观测，再调用 `reset(done_ids)`，否则下一次 step 报错 |

`info` 中的评分是环境诊断信息，训练适配器不要将隐藏评分详情或脚本专家状态拼入 policy 观测。
computer 动作与本体状态顺序为字符臂 0 的 9 维、修饰键臂 1 的 9 维。
相机分辨率由 `VectorWorkbench(..., resolution=(W,H))` 指定；提高分辨率可改善屏幕小字可读性。
脚本入口保存的轨迹是物理诊断记录，视频每 4 个控制步采一帧，不是已对齐的完整视觉 BC 数据集。

并行隔离的具体边界：

- **一环境一 VM**：只共享不可变 kernel / 基础 initramfs；独立 RAM、题目 overlay、episode UUID、
  QMP Unix socket、framebuffer 临时文件和串口日志，没有共享可写磁盘或固定 TCP 端口。
- VM I/O 使用最多 8 个线程并发，batch 有完成屏障；step / reset / close 串行，单 VM 的 QMP 也有锁。
  异步 QMP 事件不会被误作命令回复。某 VM 失败会使 batch 报错，不读取旧分数继续训练。
- guest 平时暂停，每个物理 HID 边沿只恢复短执行窗口；Enter 释放后等待该次评分完成。
  这是交互事件驱动时钟，不保证 QEMU 与 PhysX 的时间一一对应，也不承诺 KVM 逐位确定性。
- reset 销毁并重启对应 VM，更新 episode UUID、清空键状态与评分；故障清理逐个回收所有自有进程。
  这是优先保证正确性的冷启动 reset，不是快照优化。
- `qemu_child.py` 在独立解释器中设置 Linux parent-death signal 后 exec QEMU，避免在线程池使用
  不安全的 `preexec_fn`。创建 VM 的 worker / 管理线程死亡时，QEMU 收到 SIGTERM。
  强制杀死 worker 时 Python finally 不会运行，临时目录可能残留；节点应由正常的临时目录回收机制清理。
- 多 worker 用 **spawn**，在 worker 内创建 SimulationApp / VectorWorkbench / VmPool。
  不允许 fork 已启动的 Isaac / QEMU 对象；所有权校验会拒绝跨 PID 操作。
  每个 worker / rank 必须提供不同 `output` 目录（建议路径中带 rank 和 run ID）。

双臂版本已实测单 World 的 4 个编程环境（8 条机械臂、4 个独立 QEMU），四道题全部成功；
按住组合键时局部 reset、物理事件与 HID 一致性及两臂同时接触检查均通过。
另有 2 个独立 spawn worker 共 4 个真实 QEMU 的隔离与清理测试。各版本结果见下方测试记录。
这不代表千级 VM 的训练吞吐已验证。每 guest 配置 1 vCPU / 512 MiB；大批量时需为 VM、相机、
Isaac 线程和重启成本做资源预算。

## 文件与证据

- `scene.py`：机器人、弹簧键盘、笔尖、画布、材质、相机与接触采样。
- `vm.py`：QMP、Linux guest 生命周期、并行 VM pool 与局部 reset。
- `qemu_child.py`：Linux worker 意外退出时终止子 VM。
- `tasks.py` / `guest_task.py`：题库与 guest 内评分。
- `logic.py`：按键迟滞、笔迹和 IoU；`policy.py`：接触驱动的 IK 示范。
- `vector_env.py`：批量接口；`run.py`：CLI、视频、轨迹与结果检查。

每个环境输出 `before.png`、`after.png`、`rollout.mp4`、`trajectory.npz`、`result.json`，
另有绘画 `reference.png` / `canvas.png` 或编程 `screen.png`。根 `result.json` 汇总成功与局部 reset 检查。
编程加 `--record-guest-screen` 会输出 `guest-screen.mp4`，按与机器人录像相同的时刻直接采集
QEMU 显示，包含输入、提交与判题结果；录制为 15 fps，对应仿真时间，不是运行耗时。
完整验证记录见 [test_workbench.md](../../tests/isaac_workbench/test_workbench.md)，
复现入口为同目录的 `.sh`。

## 参考

实现前检查了 [NVIDIA SO-101 typing](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/06_so101_typing/so101_typing_policy.html)、
[RoboPianist](https://github.com/google-research/robopianist)、
[Spline-FRIDA](https://lawrencedchen.com/splinefrida/) 与 [IMPASTO](https://impasto-robopainting.github.io/)。
SO-101 示例需要其 Newton 补丁运行时，RoboPianist 使用 MuJoCo，本例在现有 Isaac PhysX 环境独立实现。
绘画项目用于界定任务与笔迹模型，并未声称复现它们的学习算法。
接触读取参考 [Isaac Lab issue #7613](https://github.com/isaac-sim/IsaacLab/issues/7613)；
VM 桥接依据 [QEMU QMP](https://www.qemu.org/docs/master/interop/qemu-qmp-ref.html)。
进程退出语义依据 [Linux PR_SET_PDEATHSIG](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html)。
