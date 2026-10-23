# 原生 π0.5 PPO、导出与恢复

状态：PASS，2026-10-23，H20 GPU0 真实 step100 模型测试。

脚本使用真实模型，检查不同执行长度的旧概率重算、ratio≈1、PPO 有限梯度及权重变化、全部 VLM
参数冻结、SFT 反传、FP32 master/optimizer 保存恢复、CPU/CUDA/numpy RNG 精确恢复、原生 BF16
导出重载推理。该测试通过也不代表物理任务成功或迁移有效。

先核对配置资源和 GPU 空闲、暂停本轮 dguard，然后设置 `MY_DFS`、`POLICY_CONFIG`、`TEST_OUTPUT`、
`POLICY_GPU`，执行 `bash tests/multicolor_transfer/test_policy.sh`。
test 输出必须位于当前 learner 的持久化 DFS，且使用新目录。

Carrot 基线 `ff9e882` 加当前未提交修改；RLinf commit `be1b0a4ff04955c22a0f294f9149606392e0d4e5`。
实际镜像 tag：未记录。Python 环境 `/opt/venvs/carrot`，PyTorch 2.11.0+cu128。
MCP task：`8ec98b1f-0017`；输出：
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/tests/carrot/multicolor_transfer/policy-20261023-01/result.json`。
logprob 最大误差 3.814697e-6，KL 6.7055e-8，clip fraction=0；SFT loss=0.283283；
单次 PPO 更新梯度范数（裁剪前）99.149，峰值分配显存 19.97GiB。
6 次实际控制转移、1 次更新，FP32 参数改变，骨干无梯度；保存、恢复 RNG/optimizer、导出重载均通过。
运行时 learner.py SHA256 `09fce2fe0f9a8c781e0c3e1bb4e9f3935057fbd6bc99e1874047ba59569f1c8a`，
pi05.py `024f3d77aa183a9443705fa38a90f45a1af9adcf1e667b859a07c7aba27223a0`，
test_policy.py `8958b9240ad9d4b0504bf29891c83752608cd3c4d7ba0271638a9747ca607f41`。
后续仅增加测试进度打印；这项通过不代表正式训练和迁移评测已完成。

2026-10-23 复测 `8ec98b1f-0024`：PASS，exit0，153s；新增跨次resume保留SFT对照来源的断言。
输出 `policy-20261023-02/result.json` 与 `run.log`，同一H20 DFS测试根目录。
数值指标与首轮一致；learner.py SHA `38eaf29489b65cf8bba05ddc52cb457e48f610fe9a960aa660a7838af34bb9bc`，
test_policy.py SHA `a3a0f3fbde9d221eb2e3b3508bf55e68d4a1500286124bcf32268021c8d0f6ce`。

第三轮 `8ec98b1f-0028`：PASS，exit0；改用与官方pi05_libero相同的双相机变换。
第三槽mask关闭，固定图像/文字/seed而改变8D状态后，动作逐元素完全相同（state_invariance=PASS）。
再次通过PPO/SFT反传、导出和精确恢复，old logprob误差7.45e-8，SFT loss0.288753，peak19.97GiB。
证据：H20同根目录`policy-20261023-03/result.json`、`run.log`及checkpoint。
learner SHA `30a91642de9b877f50e8d6d17c5b86d2b6f4164110c5ffed5f39904d7c47efa2`；
test_policy SHA `9372353077bfb54448918002d4224ceda3e7489b8745f45c3dd7472194bd2c4b`。

批量配置改动待复测：H20 真模型 micro=8，32 个观测分块推理/价值计算；256 个 rollout samples 完成一次 GBS=256 PPO 更新和一次 GBS=256 SFT 更新；拒绝不足批量的 PPO，预算尾部丢弃并记账。

第四轮 `8ec98b1f-0039`：PASS/exit0，220s，H20 GPU1，真实GBS256/micro8。
256个rollout samples累积32次后只做1次optimizer step（768控制转移），old logprob最大误差3.8147e-6。
真SFT入口同样完成GBS256更新，samples_seen=256，loss0.12818；32观测推理与价值计算分块通过。
不足rollout拒绝更新，32样本尾部丢弃且update计数不变；FP32/optimizer/RNG恢复、BF16导出重载和状态不变性通过。
证据：同H20测试根目录policy-20261023-04/result.json、sft.jsonl、run.log、checkpoint。
