# 逐篇研究问题与方法摘要：执行目录

任务：覆盖 new-paper-metadata.jsonl 的全部 9659 条来源记录，先去重、补摘要，再由 GPT-6.1 Sol（先 medium 校准，后依用户要求 high）逐篇提取中文 problem/solution，主代理负责校准、抽检与问题地图。

已于 2026-10-13 完成本轮全部 205 批提取和质量检查。9659 条来源记录映射为 9073 个保守去重条目：7551 条摘要支持、641 条范围外候选、881 条信息不足。待提取、必审待复核和未解决返工均为 0。

最终入口：[HTML 阅读页](../problem-map.html) · [问题地图](../problem-map.md) · [逐篇结果](../paper-summaries.jsonl) · [验收记录](final-validation.json)。问题地图包含 12 大类、各类 3–5 篇代表论文，共 57 个代表条目。

HTML 阅读页支持分类切换、代表论文全文搜索和独立论文卡片，将问题、方法步骤与实验边界分区展示。页面为单文件，可离线打开；在仓库根目录执行 `python3 docs/paper/summaries/tools/build_html.py` 可重新生成，与 Markdown 阅读版共用已复核的代表论文解释。

问题地图的阅读版将代表论文展开为完整段落，使用 `representative-explanations.json` 保存解释，使用 `map-editorial.json` 保存分类说明和术语表。解释绑定原摘要结果的哈希；修改原记录后需要重新核对。`tools/build_map.py` 从这些文件生成问题地图，避免重新生成时覆盖可读性修改。逐篇 JSONL 仍保留原来的紧凑结构化记录，不能视为全部条目均已完成详细改写。

- `canonical-inputs.jsonl`：保守去重后的输入与摘要证据。
- `source-ledger.jsonl`：每条原始来源记录到论文条目的映射。
- `calibration/`：3 组各 10 篇校准输入、输出与报告；27 条有摘要结果可复用。
- `inputs/`、`batches/`：每批最多 40 篇输入和逐篇语义总结。
- `reviews/`：主代理对全部校准样本、每批至少 2 篇及不确定项的复核记录。
- `batch-manifest.json`：批次、分工、输入 ID 和 SHA-256；批次结果先校验再合并。
- `source-cache/`：补查原始来源的本地缓存，不是独立确认的录用证据。
- `evidence-overrides.jsonl`：主代理核读后登记的补充证据；保留原始输入与哈希，正文方法概述与原摘要分别记录。
- `INSTRUCTIONS.md`：固定分类与总结规范。

允许三名子代理并行；各自只写分配批次。没有摘要的条目必须留空解决方法；不得用关键词模板生成看似逐篇阅读的总结。

## 续跑

在仓库根目录执行 `python3 docs/paper/summaries/tools/batch_io.py status` 查看有效批次数；`queue 1`、`queue 2`、`queue 3` 显示初始分工；最新调度以 `resume-state.json` 的 `worker_queues` 为准（末段有负载重分配，冻结 manifest 不改）。`review_io.py pending` 显示主代理待核读样本，不自动确认未读内容。

主代理已完成质量检查，生成上级目录的 `paper-summaries.jsonl` 和 `problem-map.md`。来源记录与保守去重条目数不同，摘要支持、信息不足、不相关分别统计。

`python3 docs/paper/summaries/tools/merge.py` 只检查当前覆盖和复核进度；只有全部批次、必审条目及返工都完成，`--publish` 才会生成最终逐篇清单。抽检发现问题后的整批复核请求与处理依据保存在 `reviews/rework-requests.jsonl`。审核记录绑定结果哈希，修改后的必审摘要会重新进入待审队列。
