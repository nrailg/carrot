import json,re,time,urllib.request,urllib.parse,xml.etree.ElementTree as ET,concurrent.futures,html
from pathlib import Path
R=Path(__file__).resolve().parents[1];CACHE=R/'source-cache';CACHE.mkdir(exist_ok=True);ns={'a':'http://www.w3.org/2005/Atom','o':'http://a9.com/-/spec/opensearch/1.1/','x':'http://arxiv.org/schemas/atom'}
rows=[json.loads(l) for l in (R/'canonical-inputs.jsonl').read_text().splitlines()];rss=[r for r in rows if not r['abstract'] and r['paper_id'].startswith('rss:')];ra={};log=[]
def fetchrss(r):
 try:
  path=CACHE/(r['paper_id'].replace(':','-')+'.html')
  if path.exists():s=path.read_text()
  else:s=urllib.request.urlopen(r['url'],timeout=25).read().decode();path.write_text(s)
  m=re.search(r'<b[^>]*>\s*Abstract:\s*</b>(.*?)</p>',s,re.S|re.I)
  a=html.unescape(re.sub('<[^>]*>',' ',m.group(1))) if m else ''
  return r['paper_id'],{'abstract':' '.join(a.split()),'url':r['url'],'status':'ok' if a else 'abstract_not_found'}
 except Exception as e:return r['paper_id'],{'abstract':'','url':r['url'],'status':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for k,v in ex.map(fetchrss,rss):ra[k]=v
(R/'rss-abstracts.json').write_text(json.dumps(ra,ensure_ascii=False,indent=2));print('RSS fetched',len(ra),'abstracts',sum(bool(v['abstract']) for v in ra.values()),flush=True)
missing=[r for r in rows if not r['abstract'] and not ra.get(r['paper_id'],{}).get('abstract')];extra={};aud=[]
for n in range(0,len(missing),10):
 group=missing[n:n+10];query=' OR '.join('ti:"'+r['title'].replace('"',' ').replace('\\',' ')+'"' for r in group);params={'search_query':query,'start':0,'max_results':100,'sortBy':'relevance'};dest=CACHE/f'title-query-{n//10:03d}.xml'
 try:
  if dest.exists():data=dest.read_bytes()
  else:
   time.sleep(3.2);data=urllib.request.urlopen('https://export.arxiv.org/api/query?'+urllib.parse.urlencode(params),timeout=35).read();dest.write_bytes(data)
  root=ET.fromstring(data);total=int(root.findtext('o:totalResults','0',ns));es=root.findall('a:entry',ns)
  for e in es:
   def val(k):return ' '.join(e.findtext(k,'',ns).split())
   aid=re.sub(r'v\d+$','',val('a:id').split('/abs/')[-1])
   extra[aid]={'arxiv_id':aid,'title':val('a:title'),'authors':[a.findtext('a:name','',ns) for a in e.findall('a:author',ns)],'abstract':val('a:summary'),'url':'https://arxiv.org/abs/'+aid,'published':val('a:published')}
  aud.append({'input_ids':[r['paper_id'] for r in group],'query':query,'total':total,'returned':len(es),'status':'ok' if total<=100 else 'truncated_candidates'})
 except Exception as e:aud.append({'input_ids':[r['paper_id'] for r in group],'query':query,'status':str(e)})
 (R/'extra-arxiv-records.json').write_text(json.dumps(list(extra.values()),ensure_ascii=False));(R/'source-enrichment-audit.json').write_text(json.dumps(aud,ensure_ascii=False,indent=2))
 if n%100==0:print('Title lookup',n+len(group),'/',len(missing),'returned_unique',len(extra),flush=True)
print('DONE title lookups',len(missing),'returned',len(extra),flush=True)
