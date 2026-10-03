# SO101拟合消融控制契约

状态：PASS（2026-10-03）

- 检验随机recipe loss与生产noise/t/32维mask objective逐元素一致。
- 固定noise必须跨batch/调用/验证一致，t仍随机；无视觉必须关闭image mask并保持原样本不变。
- 命令：确认MY_DFS后bash tests/pi_05/test_so101_fit_controls.sh。
- 预期3项CPU测试通过；不加载模型或操作机器人。
- Carrot main d4999bb加未提交recipe变更；Docker tag和上游源码commit未记录。

## 实际结果

- Gemini session af3ae39e；MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu。
- 3 passed in 8.51s，Python3.12.13/pytest9.1.1/torch2.11.0+cu128。
- 首次执行因测试fixture漏传image_keys失败；补全fixture后重跑三项全部通过，无生产代码改动。
- 原生产objective一致性逐元素零容差通过；固定noise两次t不同；所有image mask关闭且原样本未改。


## 2026-10-03提交前回归

- 状态：PASS；Gemini session d9b1a9fa，组合检查task d9b1a9fa-0209 / exit0。
- 实际命令：确认当前MY_DFS后，`bash tests/pi_05/test_so101_fit_controls.sh`。
- **3 passed in 8.62s**；本次组合回归共7项通过，仍是CPU契约测试，不加载模型或操作机器人。
- Ruff、shell语法检查通过，8个Mac/GPU待提交源码SHA256一致。
- 基于Carrot d4999bbe6557c2b91273dbe3abbd229532882d66，当前分支whyRecipeFailed加本轮未提交文件。
  提交前仅recipe import/行宽及type hints/override整理，已完成实验继续以运行时源码快照为准。
- Python3.12.13、pytest9.1.1、torch2.11.0+cu128、lerobot0.6.1；
  Docker image tag、已安装上游源码git commit未记录。
- 本次真实stdout见 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_fit_validation_commit_20261003_d9b1a9fa_0209/precommit_checks.log`，
  `provenance.json` 和 `source_hashes.json` 保存实际版本及源码证据。
