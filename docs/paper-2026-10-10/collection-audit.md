# 检索与质量记录

快照：2026-10-12。完整来源统计见 [JSON](new-paper-sources.json)。本文件区分“查询范围内完整”和“领域全量”；后者本轮未能保证。

## 1. arXiv API 与主池

使用公开 [arXiv API](https://info.arxiv.org/help/api/user-manual.html)，请求地址 `https://export.arxiv.org/api/query`，返回 Atom XML，不需要抓搜索结果网页。每页 500 条，`start` 递增，`sortBy=submittedDate`，`sortOrder=descending`；请求串行，网络请求间隔至少 3.1 秒。保留 XML 原始缓存供本次校验。

```text
cat:cs.RO AND submittedDate:[202607120000 TO 202610122359]
```

查询按 API 的 GMT/UTC 日期执行。取回后逐条检查 `published`（首次提交），保留 `updated` 但不拿更新日期当新论文。分类包括主分类与交叉分类。

主池校验结果：10 页，9 页各 500 条、末页 350 条；每页报告总数均为 4850；分页起点连续；4850 条、4850 唯一 ID；全体日期合法且单调降序；与落盘元数据 ID 集合一致。最早 `2026-07-12T03:55:51Z`，最晚 `2026-10-09T17:59:06Z`。这是本次 API 查询快照的一致性证据，无法排除上游索引延迟或后续分类变更。

## 2. 跨分类候选

第一组按标题查询，不限于 cs.CV/cs.LG/cs.AI，扩展到其他分类以提高召回：

```text
(ti:robot OR ti:robotic OR ti:robotics OR ti:"vision-language-action" OR ti:embodied OR ti:manipulation OR ti:locomotion OR ti:"world model") ANDNOT cat:cs.RO AND submittedDate:[202607120000 TO 202610122359]
```

该组合查询实际返回了不符合预期日期/分类条件的结果，原因未确定，不依赖服务端过滤正确性。API 报告 28831 条；顺序取回前 5000 条，最早已到 `2026-01-20T06:12:53Z`，跨过下界后停止。已核返回序列日期降序；本地重新过滤 `published` 和分类后保留 719 个唯一 ID。不能把 API 原始总数写成相关论文数，也不能把仅标题检索称为跨分类全量。

第二组补查摘要，分别完整分页请求，之后本地排除 cs.RO 并检查日期：

| 查询（各自追加相同 submittedDate 条件） | API 总数 | 实取条数 |
| --- | --- | --- |
| `abs:robot` | 3703 | 3703 |
| `abs:robotic` | 3703 | 3703 |
| `abs:embodied` | 1048 | 1048 |
| `abs:"vision-language-action"` | 677 | 677 |
| `abs:locomotion` | 295 | 295 |
| `abs:"autonomous driving"` | 430 | 430 |

摘要查询去重、去 cs.RO 后为 1260 条；与标题候选合并得 1692 条。robot 与 robotic 两次查询结果计数相同，仍按 ID 去重，不重复相加。候选可能涉及非机器人领域的 world model、manipulation、embodied cognition 等，原样保留并标为待筛选。关键词、英文表述、摘要未提及机器人等因素仍会漏检；未采用 Semantic Scholar/OpenAlex 引用图做二次全域补全。

## 3. 会议来源与时间边界

| 会议 | 举行时间与范围决定 | 获取结果 | 来源与清单 |
| --- | --- | --- | --- |
| RSS 2026 | 07-13—07-17，窗口内，全收官方目录 | 210 条 | [官方](https://roboticsconference.org/2026/program/papers/) · [清单](catalog/rss-2026.md) |
| IROS 2026 | 09-27—10-01，窗口内，目标全收 | 官方 Paper Index 403；1933 个二手程序条目；另有 494 条作者会议备注，不等于录用数 | [官方入口](https://2026.ieee-iros.org/program/paper_index/) · [日期旁证](https://robotics.gatech.edu/gt-iros-2026) · [候选](catalog/iros-2026-candidates.md) · [作者备注](catalog/iros-2026-author-reported.md) |
| ECCV 2026 | 09-08—09-12，窗口内，扫描整表后筛相关候选 | 2997 个展示事件 → 2834 个标题与作者组 → 291 个候选 | [官方目录](https://eccv.ecva.net/virtual/2026/papers.html) · [日期](https://eccv.ecva.net/Conferences/2026/Venues) · [清单](catalog/eccv-2026.md) |
| CoRL 2026 | 11-09—11-12，未来会议，单列已公开论文补充 | 本次未取得完整官方名单；115 条作者会议备注，不等于录用数 | [官网与日期](https://2026.corl.org/) · [Program](https://www.corl.org/program) · [作者备注](catalog/corl-2026-author-reported.md) |
| NeurIPS 2026 | 12 月，未来会议，单列已公开目录补充 | 9242 个事件 → 6347 个标题与作者组 → 368 个候选；367 条 decision 含 Accept，1 条未标明 | [官方目录](https://neurips.cc/virtual/2026/papers.html) · [日期](https://neurips.cc/Conferences/2026/Dates) · [清单](catalog/neurips-2026.md) |
| ICRA 2026 | 06-01—06-05，窗口外 | 已核日期，不整场纳入 | [官方日期](https://2026.ieee-icra.org/media/press/) |
| ICML 2026 | 07-06—07-11，截止在窗口前一天 | 已核日期，不整场纳入 | [官网与日期](https://icml.cc/Conferences/2026) |
| CVPR 2026 | 06-03—06-07，窗口外 | 已核日期，不整场纳入 | [官方日期](https://cvpr.thecvf.com/Conferences/2026/News/Media_Reg_Open) |
| ICLR 2026 | 04-23—04-27，窗口外 | 已核日期，不整场纳入 | [官网与日期](https://iclr.cc/Conferences/2026/index.html) |

官方可取数据端点：

- [RSS accepted papers](https://roboticsconference.org/2026/program/papers/)：删除 HTML 注释后提取行，避免注释中的旧表格重复计数。
- [ECCV 官方 JSON](https://eccv.ecva.net/static/virtual/data/eccv-2026-orals-posters.json)：2997 个事件，标题与完整作者集合去重后 2834 组。
- [NeurIPS 官方 JSON](https://neurips.cc/static/virtual/data/neurips-2026-orals-posters.json) 与 [摘要 JSON](https://neurips.cc/static/virtual/data/neurips-2026-abstracts.json)：9242 个展示事件，标题与完整作者集合去重后 6347 组。`sourceid` 跨 track 可能重号，不能直接按它全局去重。
- [IROS 社区候选来源](https://gisbi-kim.github.io/iros2026-explorer/output/iros2026_explorer.html)：解析页面内 papers-data JSON。二手程序数据 1933 条全部保留；未把目录条数当作 contributed paper 录用数。
- [CoRL Program](https://www.corl.org/program)：本次读取未取得完整论文表。OpenReview notes 查询返回 403，未获得完整名单。

综合会议入选规则：标题或官方关键词命中机器人/具身/VLA/SLAM/运动/灵巧操作/自动驾驶/遥操作/触觉/视觉里程计/运动规划等词；或匹配到 cs.RO 主池；NeurIPS 另以摘要命中补充。ECCV 未取得全体摘要，因此使用标题与官方关键词，会漏掉不在这些字段显式出现的相关工作。

NeurIPS 368 候选中 177 条仅摘要命中，全部明确标为待人工筛选；不能将某篇在实验背景提到 robot 等同于以机器人为研究主题。保留 Conference、Evaluations and Datasets、Position Paper 等 track。1 条 decision 未标明，不称其为已确认录用。

作者会议备注补查查询如下（不限制首次提交日期，以覆盖会议旧稿）：

```text
(co:"IROS 2026" OR co:"CoRL 2026" OR co:"IROS2026" OR co:"CoRL2026" OR jr:"IROS 2026" OR jr:"CoRL 2026")
```

API 返回 607 个唯一 ID，与其他池合并后按会议名及年份解析：IROS 494、CoRL 115 条会议提及（可能跨源重叠）。证据分为作者自报录用/发表、仅会议提及、投稿未确认、含 workshop；这些状态不是主会官方确认。未来会议的公开论文作为补充，不纳入“近三个月已经举行的会议”。

## 4. 去重、关联与分类

- arXiv 按去掉 vN 后的 ID 去重；保留首次提交和更新时间。不同 arXiv ID 不仅凭相似标题合并。
- 会议展示记录按 Unicode 规范化、去标点的完整标题＋完整作者集合分组。若作者元数据不同，保守保留为不同组，因此仍可能存在少量未合并记录。
- 官方会议与 arXiv 的关联需规范化完整标题一致、且至少一个完整作者姓名相同；多候选则不自动关联。RSS 作者从官方列表解析，因此关联率不等于目录覆盖率。
- IROS 二手目录仅显示同标题 arXiv 候选链接，未据此提升为官方确认。
- JSONL 的 9659 行是来源记录，不是跨来源唯一论文数。arXiv 记录 6857 条，包括窗口主池、跨分类候选与会议旧稿；其余是官方目录或二手候选来源记录。
- 主题标签采用关键词规则，标题优先，标题无命中时回退摘要；允许多标签。分主题文件仅索引 cs.RO 主池，会议旧稿和跨分类候选在各自完整目录中。
- 不保存全文或整段摘要至仓库；结构化文件保留书目信息、来源、作者备注与证据状态。未逐篇阅读 PDF，也未核验代码、权重和复现成本。

## 5. 明确未覆盖与下一步

1. IROS 官方全表、CoRL 官方全表仍缺；后续优先补官方来源并核验旧候选。
2. 仅通过关键词提取的跨分类和综合会议候选需要人工筛选；未保证所有相关论文都被找出。
3. 不系统穷举期刊、专项会议及 workshop；不系统纳入旧论文在窗口内仅更新 v2/v3 的情况。
4. 没有逐篇确定首次公开时间。会议条目中未匹配 arXiv 的日期写“未匹配”，不猜测。
5. 当前日期窗口采用 UTC，不是 Asia/Shanghai 零点；若后续需严格按北京时间边界，可补边界两天。

## 6. 可复核性

查询式、每页计数、原始源文件大小和 SHA-256 保存于 `new-paper-sources.json`。本次抓取缓存与执行脚本位于 `/tmp/carrot-robotics-papers-20261012`，是可被系统清理的临时文件；持久化目录与元数据保存在本目录。远端结果日后可能变化，重跑不保证得到同一快照。

## 7. 验收记录

- cs.RO：10 页连续，API 总数、实取条数与唯一 ID 数均为 4850；日期、排序、分类、落盘 ID 集合检查通过。
- 六组摘要查询：各自 API 总数与实取条数一致；本地去重、日期与排除 cs.RO 的检查通过。
- 25 个目录：标注条目数与实际表格行数逐个一致；9659 个来源记录 ID 无重复。
- 全部本地 Markdown 链接可解析；文件无尾随空白；`git diff --check` 未报告问题。本轮只有新文档，未运行项目代码测试。
- 重新打开四个月各一篇 arXiv 页面，标题与保存结果相符：[2607.18580](https://arxiv.org/abs/2607.18580)、[2608.15437](https://arxiv.org/abs/2608.15437)、[2609.22888](https://arxiv.org/abs/2609.22888)、[2610.07621](https://arxiv.org/abs/2610.07621)。抽查不能代替逐篇核验。
- 验收结论：本轮广覆盖盘点及文件整理完成；IROS、CoRL 官方全表缺口与候选人工筛选仍保留，未宣称领域全量或逐篇正确分类。
