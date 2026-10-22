# 第 2 组校准报告

完成 10/10 篇逐篇语义提取：摘要支持 8 篇，与机器人不相关 1 篇，信息不足 1 篇。

- arxiv:2610.09048 研究金融市场表征，无机器人或具身应用，按规范排除，但保留问题与方法总结。
- IROS26-4091 仅有标题，问题、方法和主分类均为空，需由主代理安排摘要补查。
- arxiv:2610.11505 的摘要未解释标题所称去噪信念学习和对抗运动先验，因此只记录摘要明确描述的循环网络、潜变量推断及强化学习。
- ManiUnit 与 FRPR 均以诊断和评测为主要贡献，主分类为 evaluation_understanding；低延迟无地图ESDF工作以查询计算开销为主要困难，主分类为 efficiency。

已校验 JSON 解析、10 条数量、输入输出 ID 顺序、完整字段、分类合法性、标签数量、证据字段原样复制，以及缺摘要条目的空值要求，均通过。

输出路径：`/Users/wujunyu/work/carrot/docs/paper/summaries/calibration/output-2.jsonl`。
报告路径：`/Users/wujunyu/work/carrot/docs/paper/summaries/calibration/output-2.report.md`。
