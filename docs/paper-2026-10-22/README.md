# CoRL / RSS / ICRA / IROS 最新一期论文扫描

快照：**2026-10-22，Asia/Shanghai**。四个会议均采用最新公开的 **2026 届**。CoRL 2026 将于 11 月 9—12 日举行，使用已公开的录用与展示程序；RSS、ICRA、IROS 采用已举行届次的公开目录。

已完整提取本次来源快照中的 **5,602 条主目录论文**，另外单列 **271 条 Late-Breaking Results**，总计 **5,873 条会议记录**。主目录包含会议技术程序中的期刊展示论文，不能用该数量计算 contributed-paper 录用率。按会议分别计数，同一工作跨会议展示时不跨会议合并。

| 会议 | 主目录 | AI 相关（主目录） | AI 候选（主目录） | LBR 补充 | 有摘要 / 全部记录 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoRL 2026 | 687 | 687 | 0 | 0 | 687/687 |
| RSS 2026 | 210 | 164 | 6 | 0 | 210/210 |
| ICRA 2026 | 2776 | 1624 | 63 | 122 | 2761/2898 |
| IROS 2026 | 1929 | 1162 | 51 | 149 | 2058/2078 |

含 LBR 的总筛选结果：**AI 相关 3,739 条，AI 候选 130 条，未标记 2,004 条**。“未标记”表示没有足够的明确证据，不代表已证明与 AI 无关。

## 阅读入口

- [完整目录：全部论文、作者与链接](all-papers.md)
- [AI 相关论文：逐条方向标签与纳入依据](ai-papers.md)
- [AI 候选：需要进一步核对的条目](ai-candidates.md)
- [AI 阅读笔记与优先阅读建议](ai-notes.md)
- [HTML 总览](index.html) · [CoRL](corl.html) · [RSS](rss.html) · [ICRA](icra.html) · [IROS](iros.html)：四个会议各有独立页面，支持 AI 状态、方向、主目录 / LBR 筛选以及全文元数据搜索。
- 按会议查看：[CoRL](catalog/corl-2026.md) · [RSS](catalog/rss-2026.md) · [ICRA](catalog/icra-2026.md) · [IROS](catalog/iros-2026.md)
- [全部结构化记录](papers.jsonl) · [AI 相关 JSONL](ai-papers.jsonl) · [AI 候选 JSONL](ai-candidates.jsonl)
- [来源与完整性审计](collection-audit.md) · [机器可读审计](collection-audit.json) · [来源快照清单与 SHA-256](source-manifest.json)

## AI 筛选口径

AI 相关包括：VLA / VLM / LLM / 基础模型；强化学习与奖励学习；模仿学习与动作策略；扩散 / 流匹配 / 生成模型；世界模型；学习式感知、神经表示与控制；表征学习和迁移；符号 AI、POMDP 等认知 / 不确定性推理；明确面向这些方法的数据、评测与基础设施。硬件论文若明确使用学习方法或服务于机器人学习评测，也可纳入。

普通 MPC、轨迹数值优化、传统状态估计、机械设计与传感器制造，不能仅因用于机器人就标记为 AI。迭代学习控制、无学习的模型驱动扩散优化等列为边界项；会保留人工判断理由。GP 回归等统计学习方法采用较宽口径。

对全部记录运行可复现规则，检查标题、作者关键词和可得摘要，并保留命中字段及方法词。标题 / 关键词明确命中、或摘要明确描述本文 AI 方法时，列入 AI 相关；仅有背景或基线线索时列候选。CoRL 的研究与配套系统按机器人学习会议范围整体纳入，具体方法标签仍逐篇提取。人工复核记录保存在 [review-overrides.json](review-overrides.json)。这是**全目录元数据与摘要扫描，加人工复核**，没有逐篇精读 PDF；AI 分类仍可能有误报和漏报。

方向为多标签，一个方向计数不代表独立论文数。标题、作者关键词标签与仅摘要出现的附加标签在 JSONL 中分别保留，避免把背景提及直接视为论文核心。

## 来源边界

- **CoRL**：官方网页内嵌程序按 OpenReview ID 去重，780 次展示对应 687 篇。作者、摘要等补充来自 2026-10-18 的社区元数据快照，与官方 687 个 ID 全部匹配；不能将这些摘要说成此次直接从 OpenReview 读取。保留原始摘要出处与镜像来源，6 条标题版本差异以官方程序标题为准。
- **RSS**：官方 RSS XXII proceedings，210 篇；全部取得官方摘要。
- **ICRA**：PaperCept 六日日程，提取有论文标题及摘要容器的技术条目；包括 122 条 LBR。有 137 条无公开摘要，保留空值，不补写摘要。
- **IROS**：当前官方 Paper & Author Index 1,929 篇；官方 poster metadata 额外有 149 条 LBR。当前目录比社区旧快照 1,933 篇少 4 条净值，使用当前官方列表；20 篇主目录无摘要。
- 全部作者列表均已取得。没有将搜索结果中的同名文章、作者自报会议备注、被拒稿件或 workshop 投稿混入主目录；不尝试穷举各独立 workshop 的论文。

## 复现

```sh
uv run --no-project --with requests --with beautifulsoup4 python docs/paper-2026-10-22/scripts/collect.py
python3 docs/paper-2026-10-22/scripts/classify.py
uv run --no-project --with markdown python docs/paper-2026-10-22/scripts/render.py
python3 docs/paper-2026-10-22/scripts/write_notes.py
python3 docs/paper-2026-10-22/scripts/validate.py
```

默认复用 `sources/` 中本次快照，避免上游后续变化造成目录漂移。完整论文内容与 PDF 可通过逐条链接继续阅读；本目录保存元数据和公开摘要，未批量下载 PDF。

## 浏览器启动

```sh
python3 -m http.server 8769 --bind 127.0.0.1 --directory /Users/wujunyu/work/carrot/docs/paper-2026-10-22
```

启动后访问 http://127.0.0.1:8769/ 。按 Ctrl+C 停止服务。浏览只需要 Python 3，不需要安装前端依赖。
