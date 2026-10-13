"""Refresh only completed valid batches; never publish the final corpus."""
import json,re
from collections import Counter
from datetime import datetime,timezone
from batch_io import R,readlines,validate,evidence_overrides
from review_io import pending,fingerprint
canonical=readlines(R/'canonical-inputs.jsonl')
rows=readlines(R/'calibration/approved.jsonl')+readlines(R/'missing-abstract-results.jsonl')
done=[]
for b in json.loads((R/'batch-manifest.json').read_text()):
 try:validate(b['batch_id'])
 except (AssertionError,FileNotFoundError,json.JSONDecodeError):continue
 done.append(b['batch_id']);rows.extend(readlines(R/'batches'/(b['batch_id']+'.jsonl')))
idx={x['paper_id']:x for x in rows};assert len(idx)==len(rows)
reviews={}
for f in (R/'reviews').glob('batch-*.jsonl'):
 for x in readlines(f):
  if x['decision']=='accept':reviews[x['paper_id']]=x.get('summary_sha256')
pilot={x['paper_id'] for x in readlines(R/'calibration/approved.jsonl')}
merged=[];over=evidence_overrides()
for c in canonical:
 pid=c['paper_id']
 if pid not in idx:continue
 row={k:v for k,v in c.items() if k!='abstract'};row.update(idx[pid]);row.update(interim=True,abstract_available=bool(c['abstract']),individually_parent_reviewed=(pid in pilot or reviews.get(pid)==fingerprint(idx[pid])))
 if pid in over:row['supplemental_evidence']=over[pid]
 merged.append(row)
(R.parent/'paper-summaries.interim.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in merged))
counts=Counter(x['status'] for x in rows);cats=Counter(x['primary_problem'] for x in rows if x['status']=='摘要支持');todo=pending();unreviewed={x['batch_id'] for x in todo}
p=R.parent/'problem-map.interim.md';s=p.read_text();rest=s[s.index('## 初步观察'):]
ed=json.loads((R/'map-editorial.draft.json').read_text())
lines=['# 机器人论文研究问题地图：中间版','', '> 全量提取和最终验收仍在进行。3 个 GPT‑6.1 Sol High 子代理继续处理，主代理逐批抽查；本文件只收录完整且通过结构校验的批次。','',f'更新时间：{datetime.now(timezone.utc).isoformat(timespec="seconds")}。', '',f'当前收集 9,659 条来源记录，保守去重为 9,073 篇。完成 **{len(done)}/205 批**，另含 27 篇校准结果；已有 **{len(rows)-counts["信息不足"]:,}** 篇摘要或相关性判断，{counts["信息不足"]} 篇缺摘要单列，剩余 **{len(canonical)-len(rows):,}** 篇待提取。','',f'状态：摘要支持 {counts["摘要支持"]:,}；不相关 {counts["与机器人不相关"]:,}；信息不足 {counts["信息不足"]}。抽查要求已完成的批次 {len(set(done)-unreviewed)}/{len(done)}，还有 {len(todo)} 项待复核；并非每篇都由主模型逐篇复核。','', '当前批次按输入顺序推进，以下数量受已处理子集影响，不能比较全领域热度或最终占比。','', '| 主问题 | 已提取数量 |','|---|---:|']
for c in ed['categories']:lines.append(f'| {c["name"]} | {cats[c["id"]]} |')
p.write_text('\n'.join(lines)+'\n\n'+rest)
print(json.dumps(dict(completed_batches=len(done),covered=len(rows),pending=len(canonical)-len(rows),pending_review=len(todo),statuses=counts),ensure_ascii=False))
