import json,sys,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def load(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def fingerprint(row):return hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def pending():
 overrides_path=R/'evidence-overrides.jsonl'
 overrides={x['paper_id']:x for x in load(overrides_path)} if overrides_path.exists() else {}
 reviewed={}
 for f in (R/'reviews').glob('batch-*.jsonl'):
  for r in load(f):
   if r['decision']=='accept':reviewed[(r['batch_id'],r['paper_id'])]=r.get('summary_sha256')
 todo=[]
 for f in sorted((R/'batches').glob('batch-*.jsonl')):
  try:out=load(f);inp=load(R/'inputs'/f.name)
  except (json.JSONDecodeError,FileNotFoundError):continue
  if len(out)!=len(inp):continue
  ix={r['paper_id']:r for r in inp};pick={inp[0]['paper_id'],inp[len(inp)//2]['paper_id']}
  report=f.with_suffix('.report.md')
  if report.exists():
   for pid in re.findall(r'(?:arxiv:)?\d{4}\.\d+|IROS26-\d+|(?:eccv|neurips|rss):\d+',report.read_text()):
    if re.fullmatch(r'\d{4}\.\d+',pid):pid='arxiv:'+pid
    if pid in ix:pick.add(pid)
  pick.update(r['paper_id'] for r in out if r['status'] in ['信息不足','仅标题暂判'] or r.get('uncertainty_note'))
  for r in out:
   if r['paper_id'] in pick and ((f.stem,r['paper_id']) not in reviewed or reviewed[(f.stem,r['paper_id'])] not in [None,fingerprint(r)]):todo.append({'batch_id':f.stem,'input':{**ix[r['paper_id']],**overrides.get(r['paper_id'],{})},'output':r})
 return todo
if __name__=='__main__':
 if sys.argv[1]=='pending':
  todo=pending();limit=int(sys.argv[2]) if len(sys.argv)>2 else 6
  print('TOTAL PENDING',len(todo))
  for x in todo[:limit]:print(json.dumps({'batch_id':x['batch_id'],'paper_id':x['input']['paper_id'],'title':x['input']['title'],'abstract':x['input']['abstract'],'body_note':x['input'].get('body_note'),'summary':x['output']},ensure_ascii=False))
 elif sys.argv[1]=='accept':
  # Explicit reviewed IDs only; no blanket acceptance of unread records.
  batch=sys.argv[2];ids=sys.argv[3:];todo={(r['batch_id'],r['input']['paper_id']):r for r in pending()};path=R/'reviews'/f'{batch}.jsonl'
  with path.open('a') as f:
   for pid in ids:
    assert (batch,pid) in todo,(batch,pid)
    f.write(json.dumps({'batch_id':batch,'paper_id':pid,'reviewer':'parent','decision':'accept','summary_sha256':fingerprint(todo[(batch,pid)]['output']),'basis':'核读完整摘要与生成结果；问题、机制、分类及不确定标记可接受。'},ensure_ascii=False)+'\n')
  print('Reviewed',batch,len(ids))
