import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
CATS={'data_cost','generalization','perception_state','instruction_reasoning','long_horizon','physical_action','autonomous_learning','safety_recovery','collaboration','efficiency','embodiment_sensing','evaluation_understanding'}
FIELDS={'paper_id','problem','solution','primary_problem','secondary_problems','methods','task_platform','evidence_url','evidence_type','status','uncertainty_note'}
def readlines(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def evidence_overrides():
 p=R/'evidence-overrides.jsonl'
 return {x['paper_id']:x for x in readlines(p)} if p.exists() else {}
def validate(batch):
 inp=readlines(R/'inputs'/f'{batch}.jsonl');out=readlines(R/'batches'/f'{batch}.jsonl');idx={r['paper_id']:r for r in inp};assert len(out)==len(inp),(len(out),len(inp));assert set(idx)=={r['paper_id'] for r in out}
 for r in out:
  x={**idx[r['paper_id']],**evidence_overrides().get(r['paper_id'],{})};assert set(r)==FIELDS,(r['paper_id'],'fields',set(r)^FIELDS);assert r['primary_problem'] in CATS|{None};assert len(r['secondary_problems'])<=2 and len(r['secondary_problems'])==len(set(r['secondary_problems']));assert set(r['secondary_problems'])<=CATS;assert r['primary_problem'] not in r['secondary_problems'];assert len(r['methods'])<=4 and len(r['task_platform'])<=3
  assert r['status'] in ['摘要支持','仅标题暂判','信息不足','与机器人不相关'];assert r['evidence_url']==x['evidence_url'];assert r['evidence_type']==x['evidence_type']
  if not x['abstract']:assert r['problem'] is None and r['solution'] is None and r['primary_problem'] is None and r['status']=='信息不足'
  if r['status']=='摘要支持':assert r['problem'] and r['solution'] and r['primary_problem']
  if r['status']=='与机器人不相关':assert r['primary_problem'] is None
 return len(out)
if __name__=='__main__':
 cmd=sys.argv[1]
 if cmd=='read':
  for r in readlines(R/'inputs'/f'{sys.argv[2]}.jsonl'):print(json.dumps({k:r.get(k) for k in ['paper_id','title','abstract','evidence_url','evidence_type','original_tags']},ensure_ascii=False))
 elif cmd=='validate':print('PASS',sys.argv[2],validate(sys.argv[2]))
 elif cmd=='queue':
  worker=int(sys.argv[2]);m=json.loads((R/'batch-manifest.json').read_text());print('\n'.join(x['batch_id'] for x in m if x['worker']==worker and not (R/'batches'/(x['batch_id']+'.jsonl')).exists()))
 elif cmd=='status':
  m=json.loads((R/'batch-manifest.json').read_text());done=[];bad=[]
  for x in m:
   if (R/'batches'/(x['batch_id']+'.jsonl')).exists():
    try:done.append((x['batch_id'],validate(x['batch_id'])))
    except Exception as e:bad.append((x['batch_id'],str(e)))
  print(json.dumps({'total_batches':len(m),'valid_batches':len(done),'valid_papers':sum(x[1] for x in done),'bad':bad,'done':[x[0] for x in done]},ensure_ascii=False))
