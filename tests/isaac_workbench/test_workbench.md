# Isaac 机器人工作台验证

验证物理键盘 → Linux HID → guest 内编程题评分 → 场景屏幕，以及画笔接触 → 笔迹 → 参考图评分。
脚本 IK 控制器验证环境可完成性，不证明已训练的视觉策略或跨任务迁移。

## 契约与复现

- CPU / QEMU：无接触不按键；迟滞区不重复触发；抬笔不连线；越界不画；QMP 异步事件不当作回复。
- 编程题：正确表达式通过全部用例，固定答案不全通过；禁止通过提交表达式伪造评分。
- 3 个真实 guest 并行：输入、题目、评分、UUID、QMP socket、输出路径独立；只重置一个 VM 不改变其他 VM。
- 2 个 spawn worker、各 2 个 VM：资源命名无冲突；正常退出和故障路径都回收自有 QEMU。
- computer：3 个 Isaac 环境分别完成 sum_two / max_list / reverse_text；实际按键序列与示范一致，所有键释放。
- paint：2 个 Isaac 环境分别接触作画，IoU ≥ 0.72 且抬笔；保持悬空的负对照无接触、无笔迹。
- 局部 reset：其他环境的关节位置/速度、画布、步数、分数、终止状态保持原值；其他 VM PID/屏幕/episode 保持原值。
- 保存真实相机视频、before/after、轨迹、评分和 reset 证据；终止观测不被自动 reset 覆盖。

入口：[test_workbench.sh](test_workbench.sh)，断言：[test_workbench.py](test_workbench.py)。
先按当前服务器实际 mount 确认 MY_DFS，并准备 guest，设置三个必填变量：

```bash
export MY_DFS=/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu
export WORKBENCH_GUEST=${MY_DFS}/benchmarks/isaac-workbench/guest-v4
export WORKBENCH_OUTPUT=${MY_DFS}/benchmarks/isaac-workbench/final01
export WORKBENCH_GPU=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd "${MY_DFS}/work/carrot"
taskset -c 32-47 bash tests/isaac_workbench/test_workbench.sh
```

以上是本次已核实的节点路径与 CPU/GPU 选择；重跑必须使用新的 WORKBENCH_OUTPUT。
仅 CPU 测试可不设置 WORKBENCH_GUEST，真实 VM 测试会明确 skip；完整验证不能把 skip 当作通过。

## 2026-10-22 至 23 日：开发验证

初始状态 NOT RUN；完成了实现与下列分项运行。源码基线
`b93902d8f33b80747fe32ae2f4d0d59b7b8b9a36`，本功能为未提交的工作区增量。
Docker image tag、Isaac Sim / Lab 源码 git commit：未记录；使用安装包版本。

环境：Ubuntu 22.04；Isaac Sim 6.0.1.0；Isaac Lab 3.0.0b2.post1；Torch 2.11.0+cu128；
QEMU 6.2 / KVM；Linux guest 5.15.0-198-generic / Python 3.10；RTX PRO 5000 72GB。
GPU 1 运行本例，GPU 0 原有用户仿真未停止。远端源码由 Mutagen carrot-zjk1 从本地同步。

证据根目录：`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu/benchmarks/isaac-workbench/`。

| 运行 | 状态 | 结果 |
| --- | --- | --- |
| probe02 | PASS，物理负对照 | 单环境 hold 60 步，零接触、零笔迹；该早期版本相机焦距错误，画面不作为视觉证据 |
| paint01 | PASS，物理绘画 | 单环境 IoU 0.789777；同上，早期画面不作为视觉证据 |
| vector-hold01 | PASS，局部 reset | 双环境 hold 30 步，零笔迹；其他环境关节不变 |
| computer03 | PASS | 3 环境分别通过 3/3 个 guest 用例；真实键序列和 key-up 检查通过；3 次局部 reset 检查通过 |
| paint02 | PASS | 2 环境 IoU 约 0.7898 / 0.8070；真实 RGB 已人工查看，目标与笔迹可见 |
| pytest（任务 0147） | PASS | 9 passed，6.81 秒；包含 3 guest、2 worker × 2 guest、崩溃传播与清理 |

开发中暴露并修复的问题：

- QEMU 6.2 不接受暂停状态下 input-send-event；改为每个上锁的输入窗口先恢复，再输入与暂停。
- Isaac Camera 的焦距与 aperture setter 使用 stage units；修正单位后工作台与屏幕正常可见，缩短远裁剪面避免邻居环境进入观测。
- Isaac fast shutdown 会覆盖异常退出码；显式打印 traceback 并传 close(exit_code=1)，失败不能表现为成功退出。
- 杀死 QEMU 的故障注入发现 buffered QMP writer 在 close 时抛 BrokenPipe，阻断其他 VM 清理；
  改用 socket.sendall 与只读 reader，清理逐 VM 收集错误并完成全部回收。该次遗留的单个测试 VM 已按 PID 回收。

节点 Kit 启动有 telemetry / nvidia-smi 子进程 fork 超时日志；限制 CPU affinity 后仿真仍能完成。
这些原生诊断可能包含环境变量，不存档未经筛选的原始 Kit 日志；保留本例自己的结果 JSON、图像、轨迹和 guest 日志。

## 2026-10-23：最终整套复验

状态：PASS。MCP task `86826e8a-0149` exit 0，重建 guest-v4 后运行上述 `.sh`。
输出 final01/。9 个 CPU / QEMU 测试通过（5.94 秒）；三环境编程全部成功；
双环境绘画 IoU 0.7898 / 0.8070；双环境 hold 的零接触、零笔迹负断言通过。
其他环境的关节速度、画布和 episode 计数保持不变等扩展 reset 检查全部通过。
hold 的 result.json 中 success=false 表示任务未完成，是负对照预期，并非断言失败。

guest-v4 kernel SHA256：`571c9fa334e5cda706a4f8fa09782164edf0a7bd4bf949caf6f2a0892639f73f`；
initramfs SHA256：`7b127e8946acf9b9a45ad634fa4c9c219d2f7d541f74bcfa865bc00246a41b09`。

追加 worker 意外退出保护后，MCP task `86826e8a-0152` exit 0：10 个 CPU / QEMU 测试
7.03 秒通过，三环境编程全部成功，输出 final02/computer。启动前已核对本地和远端源码 SHA256。
前一轮 0150 的多进程测试因测试文件同步过程中被读取而失败；已停止修改运行代码，
flush 完成后再验证。该失败不记为功能通过。

最终检查：节点无残留 QEMU 进程；本次暂停的 dguard 已恢复启用，自动恢复定时器已取消。
本地 ruff check / format 与 bash -n 通过；编辑器 diagnostics 工具不可用，未以它作为验证依据。
视频和可视化证据已复制到本地 `outputs/isaac_workbench/final01/paint/env-000/` 与
`outputs/isaac_workbench/final02/computer/env-001/`。未修改核心训练模块，未启动学习训练。
