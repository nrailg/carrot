import json,urllib.request,re,html,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1];cache=R/'source-cache';rows=[r for r in map(json.loads,(R/'canonical-inputs.jsonl').read_text().splitlines()) if not r['abstract'] and r['paper_id'].startswith('neurips:')]
def fetch(r):
 try:
  p=cache/(r['paper_id'].replace(':','-')+'.html')
  if p.exists():s=p.read_text()
  else:s=urllib.request.urlopen(r['url'],timeout=30).read().decode();p.write_text(s)
  m=re.search(r'class="abstract-text-inner"[^>]*>(.*?)</div>',s,re.S);a=' '.join(html.unescape(re.sub('<[^>]+>',' ',m.group(1))).split()) if m else ''
  return r['paper_id'],{'abstract':a,'url':r['url'],'status':'ok' if a else 'not_found'}
 except Exception as e:return r['paper_id'],{'abstract':'','url':r['url'],'status':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:d=dict(ex.map(fetch,rows))
(R/'extra-venue-abstracts.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print('Official NeurIPS page abstracts',sum(bool(v['abstract']) for v in d.values()),'/',len(d),flush=True)
