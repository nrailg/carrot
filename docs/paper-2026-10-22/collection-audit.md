# 检索、来源与完整性审计

快照：2026-10-22（Asia/Shanghai）。本文件记录元数据扫描的范围与可验证边界，不把目录全量等同于逐篇精读。

## 范围与计数

- 主目录只接纳官方录用 / proceedings / 技术程序中的条目。
- 同一会议按来源 ID 去重；保留多次展示的 session 信息。
- 期刊展示条目计入会议程序；LBR 单列；各独立 workshop 的论文不在穷举范围。
- 不以 arXiv 作者自报会议、搜索结果或非录用的 OpenReview 投稿补主目录。

| 会议 | 主目录 | LBR | 有摘要 | 有作者 |
| --- | ---: | ---: | ---: | ---: |
| CoRL 2026 | 687 | 0 | 687 | 687 |
| RSS 2026 | 210 | 0 | 210 | 210 |
| ICRA 2026 | 2776 | 122 | 2761 | 2898 |
| IROS 2026 | 1929 | 149 | 2058 | 2078 |

## CoRL 2026

- [官方录用程序](https://2026.corl.org/program/accepted-papers) 中的论文表以 Google Sites `data-code` 内嵌 HTML 保存；单纯读取可见正文会遗漏整张表。
- 解析 780 次展示，按 OpenReview ID 去重为 687 篇；展示次数不能当论文数。
- 作者与摘要来自 [社区元数据 JSON](https://gisbi-kim.github.io/conference-explorers/corl2026/data.json)，快照标注更新于 2026-10-18。与官方 687 个 ID 一一匹配，没有额外或缺失 ID。
- OpenReview API 此次请求返回 403；因此明确标注摘要通过社区镜像获得。记录中保留 `abstract_source` 指向原始出处，`metadata_mirror` 指向此次读取的镜像。
- 6 条标题文字 / 数学排版差异如下，保留官方程序标题；其中 RARM 的差异属于实质改题，其余主要为符号空格。

| OpenReview ID | 官方程序标题 | 社区元数据标题 |
| --- | --- | --- |
| 5XKFwWrW0v | π 0.7 : a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities |
| AAn8ouEHwO | RARM: Confidence-Gated Progress Reward Modeling for RL in Manipulation | Progress That Trains: Reference-Anchored Reward Modeling for Robotic Manipulation from a Single Demonstration |
| GFxjqZVSND | M 3 P-R1: Reinforcement Learning for Large Language Model Guided Multi-Modal Motion Planning via MIP Code Generation | M3P-R1: Reinforcement Learning for Large Language Model Guided Multi-Modal Motion Planning via MIP Code Generation |
| kGLogzKtYq | GROW 2 : Grounding Which and Where for Robot Tool Use | GROW2: Grounding Which and Where for Robot Tool Use |
| qvOl1xOUtn | τ 0 -WM: A Unified Video-Action World Model for Robotic Manipulation | τ0-WM: A Unified Video-Action World Model for Robotic Manipulation |
| ShBMWDRPVu | μ 0 : A Scalable 3D Interaction-Trace World Model | μ0: A Scalable 3D Interaction-Trace World Model |

## RSS 2026

- [RSS XXII 官方 proceedings](https://roboticsproceedings.org/rss22/) 提供 210 个论文页；逐页取得官方标题、作者、摘要、PDF 链接和 DOI。
- 标题以论文详情页的 `citation_title` 为准，修正目录页默认编码带来的破折号乱码。
- 210/210 均有摘要；按 `p001` 等官方编号记录。

## ICRA 2026

- 从 [官方会议程序入口](https://2026.ieee-icra.org/Program/) 跟到 [PaperCept 六日日程](https://ras.papercept.net/conferences/conferences/ICRA26/program/)。
- 以 `.pTtl` 标题中具有 `viewAbstract` ID 的条目作为论文记录，排除 keynote、panel、chair 等活动条目。
- 六日论文条目数为 0、0、968、969、961、0，合计 2,898 个唯一 ID。去除 session 明确标为 Late Breaking Results 的 122 条后，主目录为 2,776 条。
- 全部 2,898 条标题与作者均已取得；2,761 条有公开摘要，137 条保留空值。
- PaperCept 的标点按 Windows-1252 解码；清除源文本中的非打印控制字符，保留标题原文。
- `paper_id` 是来源程序使用的标识；不额外推断它是否等于投稿编号，也不把程序数量称作 contributed-paper 录用数。

## IROS 2026

- 旧路径 `paper_index`、`contributed_talks`、`poster_sessions` 在脚本请求中返回 403；通过浏览器确认当前页面使用连字符路径。
- [当前官方 Paper & Author Index](https://2026.ieee-iros.org/program/paper-index/) 内的 `pi-data` 有 1,929 个唯一论文编号，作为主目录完整基准。
- [官方 Poster Sessions](https://2026.ieee-iros.org/program/poster-sessions/) 的 `poster-map-data` 有 2,078 条：完整覆盖主目录 1,929 个 ID，另有 149 个 ID，全部标为 Late-Breaking Results，单列保存。
- 主目录有 1,909 条摘要，缺 20 条；LBR 149 条摘要齐全。
- 当前官方目录与 [社区旧快照](https://gisbi-kim.github.io/iros2026-explorer/output/iros2026_explorer.html) 的 1,933 条不同。旧快照有而当前官方没有的编号为 `21, 244, 323, 5043, 5053`；当前官方新增 `1812`。这里只记录目录差异，不推断撤稿或缺席原因。
- 官方索引中包含期刊展示；1,929 与 contributed-paper 的录用数是不同口径。

## AI 分类与人工复核

- 全部条目逐记录运行标题、作者关键词和摘要规则；没有 PDF 的条目也不会被静默丢弃。
- 区分标题 / 关键词直接证据与摘要提及；仅背景提及的条目优先列候选。对摘要描述本文方法的明确语句另做上下文筛选。
- 人工检查了边界候选、RSS 的未标记目录和 16 篇阅读笔记；判断覆盖以 `review-overrides.json` 为准，其他条目不宣称人工逐篇审核。
- 已修正 LoRa 无线通信与 LoRA 适配混淆、cul-de-sac 与 SAC 缩写混淆；ACT 缩写使用大小写约束。
- 人工判断仍有范围取舍，例如无学习的扩散数值优化、迭代学习控制和连续时间 GP 标定。完整理由保留在条目的 `ai_basis` 中。
- AI 相关清单不等同于经过专家精读后确定的领域真值；候选与未标记条目保留，便于后续修改筛选口径。

## 可复现证据

- [source-manifest.json](source-manifest.json) 给出来源 URL、落盘路径、日期、解压后的 UTF-8 快照字节数和 SHA-256。源快照以 gzip 压缩，`collect.py` 可直接复用，无需重新联网。
- [collection-audit.json](collection-audit.json) 记录去重、跨来源 ID 差异与摘要 / 作者覆盖。
- [screening-stats.json](screening-stats.json) 保存会议级筛选与多标签统计；[review-overrides.json](review-overrides.json) 保存人工复核覆盖。
- [validation.json](validation.json) 核对唯一 ID、官方名册 ID 集合、分类子集、作者 / 摘要覆盖、Markdown 行数、来源校验和、内部链接、离线页面内嵌数据及 JavaScript 语法。
