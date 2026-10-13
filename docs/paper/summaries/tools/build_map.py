"""Render the parent-authored synthesis after full extraction and representative review."""
import json
from collections import Counter

from batch_io import R, CATS, FIELDS, readlines
from review_io import fingerprint


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def main():
    validation = json.loads((R / 'final-validation.json').read_text())
    assert validation['complete']
    papers = readlines(R.parent / 'paper-summaries.jsonl')
    index = {x['paper_id']: x for x in papers}
    editorial = json.loads((R / 'map-editorial.json').read_text())
    assert {x['id'] for x in editorial['categories']} == CATS
    reviewed = {x['paper_id']: x.get('summary_sha256') for x in readlines(R / 'reviews/representatives.jsonl') if x['decision'] == 'accept'}
    selected = [pid for section in editorial['categories'] for pid in section['representatives']]
    assert set(selected) <= set(reviewed)
    for pid in selected:
        assert reviewed[pid] == fingerprint({k: index[pid][k] for k in FIELDS}), (pid, "representative review is stale")
    statuses = Counter(x['status'] for x in papers)
    supported = [x for x in papers if x['status'] == '摘要支持']
    primary = Counter(x['primary_problem'] for x in supported)
    secondary = Counter(k for x in supported for k in x['secondary_problems'])
    assert sum(primary.values()) == len(supported)
    lines = ['# 机器人研究问题地图：摘要级综述', '', *editorial['introduction'], '',
             f'原始来源记录 **{validation["source_records"]:,}** 条，保守去重论文条目 **{len(papers):,}** 个。', '',
             '| 条目状态 | 数量 |', '|---|---:|']
    for status in ['摘要支持', '与机器人不相关', '信息不足', '仅标题暂判']:
        lines.append(f'| {status} | {statuses[status]:,} |')
    lines += ['', '主问题每篇只计一次；次问题计数允许一篇涉及多个类别。以下数量描述本次收集语料，不代表整个领域的发表占比。', '',
              '| 研究问题 | 主问题篇数 | 作为次问题篇数 |', '|---|---:|---:|']
    for section in editorial['categories']:
        lines.append(f'| {section["name"]} | {primary[section["id"]]:,} | {secondary[section["id"]]:,} |')
    lines += ['', '## 分类阅读', '', '子问题与解决路线是主代理对本轮摘要的归纳，并非互斥的正式二级标签。代表论文用于说明路线，不表示质量排名；每篇代表论文均重新核读证据。']
    for section in editorial['categories']:
        assert 3 <= len(section['representatives']) <= 5
        lines += ['', f'### {section["name"]}', '', '**子问题：**' + section['subproblems'], '', '**主要解决路线：**' + section['routes'], '',
                  '| 代表论文 | 研究问题 | 解决思路 | 证据边界 |', '|---|---|---|---|']
        for pid in section['representatives']:
            paper = index[pid]
            assert paper['status'] == '摘要支持'
            assert section['id'] in [paper['primary_problem'], *paper['secondary_problems']], (section['id'], pid)
            link = f'[{cell(paper["title"])}]({paper["evidence_url"]})'
            limitation = paper.get('uncertainty_note') or '摘要级材料；作者报告，未独立复现。'
            lines.append(f'| {link} | {cell(paper["problem"])} | {cell(paper["solution"])} | {cell(limitation)} |')
        if section.get('notes'):
            lines += ['', section['notes']]
    lines += ['', '## 交叉观察与后续阅读', '', *editorial['cross_cutting'], '',
              '## 范围与证据限制', '', *editorial['limitations'], '',
              '## 数据与复核记录', '',
              '- [逐篇中文问题与方法 JSONL](paper-summaries.jsonl)',
              '- [原始论文目录](new-paper.md)',
              '- [输入来源到论文的映射](summaries/source-ledger.jsonl)',
              '- [批次、分类规范和续跑方法](summaries/README.md)',
              '- [最终覆盖校验](summaries/final-validation.json)',
              '- [代表论文复核记录](summaries/reviews/representatives.jsonl)', '']
    (R.parent / 'problem-map.md').write_text('\n'.join(lines))
    print(f'Wrote problem-map.md: {len(papers)} papers, {len(selected)} representative placements.')


if __name__ == '__main__':
    main()
