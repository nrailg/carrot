# 自定义 Isaac 入门环境真实 GPU 验证

## 目的与预期

验证 `examples/isaac_custom_env` 的 Lab / Arena 两条入口。测试按顺序在一张空闲 GPU
运行，环境 batch 放在同一进程中。基线要求：

- 两个入门入口各执行 160 步并正常退出。
- 两个四环境合同检查：观测 `[4, 28]`、动作 `[4, 7]`、30 Hz 控制、150 步回合。
- 关节动作顺序及 `q_default + 0.25 * action` 量纲正确。
- 用实际 hand 位姿构造目标，只触发指定槽位成功；超时槽位只返回 truncated。
- 自动 reset 仅影响结束槽位，返回观测包含重新采样的目标。
- 正常目标采样下真实推进 160 步，无 NaN/Inf，发生回合结束。

构造成功目标只测试判据，不证明摆动策略能完成 reach。没有 SO101 资产测试，也没有
相机图像质量、RL learner 更新、final_observation wrapper 验证。

## 复现命令

按远程技能预检并同步后，使用已确认的 `MY_DFS`、`ISAAC_ASSET_ROOT`、`CUDA_VISIBLE_DEVICES`
和新的持久化 `CUSTOM_ENV_RUN_DIR`：

```bash
bash tests/isaac_custom_env/test_custom_env.sh
```

四个进程串行执行。结果 JSON 记录实际版本、时间、文件 SHA256 和关键断言；只有所有步骤
通过才输出 `CUSTOM_ENV_MATRIX_PASS`。后台任务输出另行持久化到同一结果目录。

## 2026-09-23 执行记录

状态：BLOCKED（连接未建立），所有 GPU 用例为 NOT RUN。

- Luna 使用用户指定 launcher 调用 `remote_connect`，返回
  `Failed to connect to remote container`；`remote_status` 显示 `No active sessions`。
- 主模型使用同一目标独立复试一次，得到相同错误。没有 session_id，不能进行会话级诊断。
- 未进行远程源码同步、DFS/版本/GPU/资产预检，未暂停 dguard，未启动仿真。
- 本地 `ruff check`、`ruff format --check`、AST 解析和 `bash -n` 通过。
- 已请求用户确认 Pod 状态/提供当前可用连接，恢复后继续同一测试事项。

Docker image：未获取；远端 Carrot commit：未获取；上游版本：未获取；
GPU/资产路径/持久化运行日志：未获取。连接凭据不写入档案。

## 2026-09-23 新节点准备

新 launcher 已于北京时间 23:20 左右连接成功。DFS 检测得到当前个人目录
`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu`。镜像标识为
`wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8`；Python 3.12.13，
Isaac Sim 6.0.1.0、Isaac Lab 3.0.0b2.post1、Arena 0.3.0。8 张 GPU 为 RTX PRO 5000，
预检时运行的 PID 971–978 属于 dguard。GPU 测试尚未启动。

本节点的个人 CephFS 起初没有完整 Isaac 资产包。devcloud 上另一块**个人 CephFS**
已有 Isaac 6.0 的 Panda、Grid 和 UIElements 子树；本轮只将这些子树定向同步到当前
`MY_DFS/isaacsim_assets/Assets/Isaac/6.0/Isaac/`，并修改示例显式从该根读取三种 USD。
资产仍只放在个人 CephFS：从已有的个人 CephFS 定向同步了 Panda、Grid、
UIElements 三个子树，没有放入节点临时盘。`rsync -ain --checksum` 对三个子树均无差异；
远端确认三个入口 USD 存在。示例与测试脚本已同步到
`$MY_DFS/work/carrot`，执行文件的本地/远端 SHA256 一致。

Luna 在物理 GPU 1 空闲后，停止本节点 dguard 巡检，并尝试启动后台测试任务
`11bf11e8-0019`。预定结果目录是
`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu/benchmarks/isaac-custom-env/20260923-153419`。
任务在 `mkdir` 阶段以退出码 1 结束，错误为：

```text
mkdir: cannot create directory '/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu/benchmarks/isaac-custom-env': Permission denied
```

独立复核：远程 MCP 命令的 UID 为 0，个人 CephFS 目录归 UID 1001、权限 755，
该节点没有 UID 1001 的 passwd 条目。挂载类型为 `fuse.dop-fuse`，启用了
`default_permissions`；进程持有 `CAP_DAC_OVERRIDE`，UID 映射也是 0→0。因此普通
Linux 目录权限 `755` **不能单独解释** root 的拒绝，可能是 FUSE/后端的附加权限策略；
目前未取得能区分具体策略的证据。
开发机侧同一路径由 UID 1001 `nrwu` 经 `fuse.ceph-fuse3` 挂载，可写入并已用于资产同步；
这与 Gemini 容器的 `fuse.dop-fuse` 挂载不同，不能把开发机写入成功直接外推到容器。
故本次是 **FAIL（启动前持久化目录权限）**；
Lab/Arena 四个 GPU 用例均为 **NOT RUN**，没有仿真成功或失败的结论。需将测试会话
配置为可按个人 CephFS 所有者身份写入，或提供已配置可写的个人 CephFS 结果目录；
取得后沿用本测试矩阵继续。未修改目录权限或把结果转存节点临时盘。
由于未进入仿真，测试前暂停的本节点 dguard 已执行
`bash /root/dguard/dguard.sh on --local` 恢复；主模型独立复核
`DGUARD_WATCH=1`、guard 与 run.py 均在运行，自动恢复计时已取消。
