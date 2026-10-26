# SO101 单关节往复运动

状态：PASS（2026-10-26，真实 Isaac Sim / PhysX 两周期验证）。

目的：加载真实 SO101 PhysX USD，运行与交互演示相同的控制循环，验证实际关节运动。
默认 ±30°、4秒周期、120Hz、960步，共两个周期。

断言：第二周期实测 shoulder_pan 正负幅度均超过0.4rad，跟踪 RMSE 小于0.15rad；
全程幅度小于35°，其余关节第二周期偏离零位小于0.05rad；基座位移小于1mm且姿态稳定。
进程必须成功退出、完整保存960帧且全部值有限。

运行前确认 GPU 空闲、按远程技能处理 dguard，并设置本次会话解析的 MY_DFS、
本地 SO101_USD 及可选 SO101_GPU，再执行：

```bash
bash tests/isaac_so101/test_motion.sh
```

本测试是单进程 Isaac Sim，不使用 Ray。标准输出保留模拟器日志，pytest 临时目录保存 motion.npz。

版本：Carrot 基线 main `5d57dee`；本次新增代码未提交。
Docker image 和上游安装 commit：未记录。实际 Isaac Sim 版本和运行证据见下。
本地已读取 NVIDIA 6.0 官方资产的顶层及 physics/base USD，确认固定根关节路径、
六个关节名称与 ±30° shoulder_pan 限位；此检查不代替 GPU 物理验证。

2026-10-26 本地检查 PASS：两个 Python 文件语法检查、ruff check、run.py --help、
test_motion.sh 的 bash -n。当时尚未运行 test_motion.py。

## 2026-10-26 GPU 验证

任务 `6adb7e1b-0478` 退出0，pytest **1 passed in 42.33s**。
Python 3.12.13、Isaac Sim 6.0.1.0、NumPy 2.3.1、pytest 9.1.1；
RTX PRO 5000 72GB Blackwell，GPU 1（UUID 前缀 `8a5b4e94`）。

- 960帧全部有限，完整两个周期。
- 第二周期实际转角：−28.6216° 到 +28.6218°。
- 第二周期 pan 跟踪 RMSE：6.2306°，反映位置 PD 的相位滞后。
- 其余关节最大零位偏差：0.6285°。
- 基座最大位移：0m；四元数绝对点积最小值约1，姿态不变。

实际源文件 SHA256：run.py `b0ab61e20df5c3e00953583dfb6a112c7ae49cea5c673b06ca0e31862c4d3950`；
test_motion.py `c17c0cd0831348ecc1e0c6c0331cc4f4b1d2492f4828edb4abd7c46f53fadef1`。
本地与远端一致。官方顶层资产 SHA256：
`04ef30434ff8bb752241ce5c6d33304ad04629b5d9290c580c4fd8b9fc89efb3`，USD composition errors 为空。

本轮 MY_DFS 为 `/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu`。
资产为其下 `isaacsim_assets/Assets/Isaac/6.0/Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd`。
证据保存在 `${MY_DFS}/tests/carrot/isaac_so101/motion-20261026-02/`：
`test.log`、`kit.log`、`metrics.json` 和 `test_so101_sweeps_with_fixed_b0/motion.npz`。

实际执行 test_motion.sh，额外设置 `SO101_GPU=1`、
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` 和
`PYTEST_ADDOPTS=--basetemp=${MY_DFS}/tests/carrot/isaac_so101/motion-20261026-02`。
启动器将 CPU affinity 限制到当前可用的前16核，并在子进程启动前移除凭据类环境变量。
复跑使用新的 basetemp 路径，先创建其父目录，避免 pytest 清理已有证据。

首次启动因 basetemp 父目录不存在而失败；第二次在默认地面联网查询资产时被主动停止。
已改为本地创建 ground plane，并限制 Kit CPU 线程数。
容器的两份 NVIDIA ICD 将每张卡枚举两次；本次仅为测试进程选择单份 ICD，未修改系统驱动文件。
通过轮仍有 Kit 辅助 nvidia-smi/telemetry 子进程启动报错，但物理循环完整完成且进程退出0。

测试后仿真进程已退出，dguard 恢复为 DGUARD_WATCH=1、无延迟恢复任务；
原 GPU 0 用户进程 PID630356 仍运行。未启动持续演示，未做截图或 GUI 交互验收。
