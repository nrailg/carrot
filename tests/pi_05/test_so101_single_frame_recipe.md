# SO101单帧recipe数据选择

状态：PASS（2026-10-01）

- 验证固定源帧、原stats保留、64虚拟样本在8rank/micro4/GAS2下能提供完整batch。
- 越界frame_index和非正repeat_count必须失败，不回退到全量数据。
- 命令：确认MY_DFS后运行`bash tests/pi_05/test_so101_single_frame_recipe.sh`。
- 预期：4项CPU测试通过，不加载真实模型或操作机器人。
- 实际源码main d4999bb加未提交recipe改动，Docker tag未记录；结果见下方。

## 2026-10-01运行结果

- Gemini session d17fba5b，命令 `export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu; bash "$MY_DFS/work/carrot/tests/pi_05/test_so101_single_frame_recipe.sh"`。
- 4 passed in 6.39s；8rank各2个micro4 batch全部为frame53，原stats引用保留，三个非法选择失败。
- Python3.12.13、pytest9.1.1、torch2.11.0+cu128、lerobot0.6.1；main d4999bbe6557c2b91273dbe3abbd229532882d66加未提交改动。
  Docker tag及已安装上游源码commit未记录。
- 真实数据预检另通过：source264帧，Subset64项全指向53，首末重复样本及源53逐字段一致，50步均有效。
  证据：单帧实验monitor/20260930T164407Z/preflight.json与mac_source_hashes.json，62个源码/recipe哈希与Mac一致。


## 2026-10-03提交前回归

- 状态：PASS；Gemini session d9b1a9fa，组合检查task d9b1a9fa-0209 / exit0。
- 实际命令：确认当前MY_DFS后，`bash tests/pi_05/test_so101_single_frame_recipe.sh`。
- **4 passed in 6.57s**；本次组合回归共7项通过，仍是CPU契约测试，不加载模型或操作机器人。
- Ruff、shell语法检查通过，8个Mac/GPU待提交源码SHA256一致。
- 基于Carrot d4999bbe6557c2b91273dbe3abbd229532882d66，当前分支whyRecipeFailed加本轮未提交文件。
  提交前仅recipe import/行宽及type hints/override整理，已完成实验继续以运行时源码快照为准。
- Python3.12.13、pytest9.1.1、torch2.11.0+cu128、lerobot0.6.1；
  Docker image tag、已安装上游源码git commit未记录。
- 本次真实stdout见 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_fit_validation_commit_20261003_d9b1a9fa_0209/precommit_checks.log`，
  `provenance.json` 和 `source_hashes.json` 保存实际版本及源码证据。
