# 多色迁移算法与数据契约

状态：PASS，2026-10-23，19 passed in 1.79s（含现有 placement 回归 4 项）。

检查 Flow-SDE 全部五个去噪位置的独立标量公式、Gaussian logprob、执行长度和七维动作 mask、
SMDP GAE 的终止/截断与 reset 边界、PPO ratio=1 梯度、分色 IoU 和离纸断线、oracle 一次性奖励、
示范动作前观测对齐及子任务 mask、三 checkpoint 配对统计、跨池显式节点放置。

执行：确认同步后，`MY_DFS=<当前会话个人 DFS> bash tests/multicolor_transfer/test_transfer.sh`。
不占用 GPU，不启动 Ray。节点选择测试只 mock 节点清单，执行真实选择逻辑。

Carrot 基线：`ff9e882` 加本次未提交修改；准确内容以运行时 SHA-256 清单为准。
RLinf 公式参考：`be1b0a4ff04955c22a0f294f9149606392e0d4e5`。
实际镜像 tag：未记录。Pro5000 launcher，Python 3.12.13，pytest 9.1.1，PyTorch 2.11.0+cu128。
执行目录：`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu/work/carrot`。
命令：`MY_DFS=/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu bash tests/multicolor_transfer/test_transfer.sh`。
同步 SHA-256：test_transfer.py `573b334345ca7cd01f37501c6e15888f38ca6acc202931f008f8303573c65e87`；
algorithms.py `71444aa41ec11c1b15626a2668d425de9347bcfeb535213211a9864afd186b03`；
placement.py `39a03e6ae6afe380bebe2a7d8c8033d3dd54c397f053d3a2cab2ebe7c0712440`；
ray_runtime.py `f693fcba14495b80baef9300abc4af78b8f320c7ea78016919dd9864f9ca09bb`。
上述为 CPU 契约测试，不构成 GPU、物理任务或跨池连通性验收。

2026-10-23 追加：加入 LIBERO 坐标、EEF 旋转、正负手指状态和增量夹爪约定，18 项全部通过。
任务 `80e65273-0206`，exit0；test_transfer.py SHA-256
`0e01ce4deca290ddab6b0f35e0e8928477c0d11a687dde79a598ff64f2ed6b4b`。

第三轮新增仿真子进程凭据环境隔离检查：`80e65273-0236`，exit0，19 passed in 1.79s。
test_transfer.py SHA `d90cf9ac7fb1798bd3680e6c6d929bd0c34066fdcdb30456208faeab7e796381`。
日志在Pro5000个人DFS的 `tests/carrot/multicolor_transfer/test-transfer-20261023-03.log`。

第四轮双相机示范数据契约回归：`80e65273-0250`，exit0，19 passed in1.80s。
日志同目录`test-transfer-20261023-04.log`；不再要求独立reference_image数组。

批量配置改动待复测：GBS=256、rollout=4096 的整除约束；变长 SFT 样本的 micro 分块梯度不变性；中断采集仅复用已提交成功示范。

全局批量/复用验收：22项PASS，1.86s，task80e65273-0276/exit0。
包含必需示范数组校验后的最终版本；日志为Pro5000个人DFS下tests/carrot/multicolor_transfer/test-transfer-20261023-06.log。
