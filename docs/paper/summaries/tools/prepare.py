import json,re,unicodedata,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/tmp/carrot-robotics-papers-20261012');DOC=ROOT.parent
raw=[json.loads(l) for l in (DOC/'new-paper-metadata.jsonl').read_text().splitlines()]
def norm(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())
def namekey(s):return ''.join(sorted(re.findall(r'[a-z]+',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower())))
def names(r):
 a=r.get('authors',[])
 if isinstance(a,str):a=[x.strip() for x in a.split(',')]
 return {namekey(x.get('name','') if isinstance(x,dict) else x) for x in a if x}
cache={}
for file in ['arxiv_records.json','arxiv_cross_records.json','arxiv_cross_filtered.json','arxiv_abstract_records.json','arxiv_venue_records.json']:
 for r in json.loads((P/file).read_text()):cache[r['arxiv_id']]=r
if (ROOT/'extra-arxiv-records.json').exists():
 for r in json.loads((ROOT/'extra-arxiv-records.json').read_text()):cache[r['arxiv_id']]=r
cachetitle=collections.defaultdict(list)
for r in cache.values():cachetitle[norm(r['title'])].append(r)
neurips=json.loads((P/'neurips_abstracts.json').read_text());eccv=json.loads((P/'eccv_abstracts.json').read_text()) if (P/'eccv_abstracts.json').exists() else {}
rssabs=json.loads((ROOT/'rss-abstracts.json').read_text()) if (ROOT/'rss-abstracts.json').exists() else {}
venuepages=json.loads((ROOT/'extra-venue-abstracts.json').read_text()) if (ROOT/'extra-venue-abstracts.json').exists() else {}
canon={};index=collections.defaultdict(list);ledger=[]
for r in raw:
 abstract='';evurl='';evtype='missing';url=r.get('url')
 if r['record_type']=='arxiv':
  abstract=cache.get(r['arxiv_id'],{}).get('abstract','');evurl=r['url'];evtype='arxiv_abstract'
 else:
  if r.get('arxiv_id') in cache:abstract=cache[r['arxiv_id']]['abstract'];evurl=cache[r['arxiv_id']]['url'];evtype='arxiv_abstract'
  if not abstract:
   pool=[x for x in cachetitle[norm(r['title'])] if names(r)&names(x)]
   if len(pool)==1:abstract=pool[0]['abstract'];evurl=pool[0]['url'];evtype='arxiv_abstract'
  if not abstract and r['record_type']=='conference':
   if r.get('venue') in ['NEURIPS 2026','ECCV 2026']:
    d=neurips if r['venue']=='NEURIPS 2026' else eccv
    for eurl in r.get('event_urls',[r['url']]):
     a=d.get(eurl.rsplit('/',1)[-1])
     if a and isinstance(a,str):abstract=a;evurl=eurl;evtype='official_abstract';break
   if r.get('venue')=='RSS 2026':
    abstract=rssabs.get(r['id'],{}).get('abstract','');evurl=r['url'];evtype='official_abstract' if abstract else 'missing'
 if not abstract and venuepages.get(r['id'],{}).get('abstract'):
  abstract=venuepages[r['id']]['abstract'];evurl=venuepages[r['id']]['url'];evtype='official_abstract'
 key=r['id'];matchrule='new_record'
 if r.get('arxiv_id') and 'arxiv:'+r['arxiv_id'] in canon and r['record_type']!='arxiv':key='arxiv:'+r['arxiv_id'];matchrule='existing_verified_arxiv_link'
 elif r['record_type']!='arxiv':
  hits=[c for c in index[norm(r['title'])] if names(r)&set(c['_author_keys'])]
  if len(hits)==1:key=hits[0]['paper_id'];matchrule='exact_normalized_title_and_full_author'
 if key not in canon:
  canon[key]={'paper_id':key,'title':r['title'],'url':url,'authors':r.get('authors',[]),'abstract':abstract,'evidence_url':evurl or url,'evidence_type':evtype,'source_ids':[],'source_types':[],'original_tags':r.get('tags',[]),'published':r.get('published'),'_author_keys':sorted(names(r))};index[norm(r['title'])].append(canon[key])
 c=canon[key];c['source_ids'].append(r['id']);c['source_types'].append(r['record_type'])
 if not c['abstract'] and abstract:c.update(abstract=abstract,evidence_url=evurl,evidence_type=evtype)
 if not c['url']:c['url']=url or (evurl if abstract else None)
 ledger.append({'source_id':r['id'],'paper_id':key,'match_rule':matchrule})
for c in canon.values():
 c['possible_duplicate_ids']=[other['paper_id'] for other in index[norm(c['title'])] if other['paper_id']!=c['paper_id']]
 c.pop('_author_keys')
 if not c['abstract']:
  c['evidence_type']='title_only'
  c['evidence_url']=c['url'] or 'https://gisbi-kim.github.io/iros2026-explorer/output/iros2026_explorer.html'
  c['evidence_scope']='collection_index' if not c['url'] else 'individual_paper'
 else:c['evidence_scope']='individual_paper'
for file,rows in [('canonical-inputs.jsonl',list(canon.values())),('source-ledger.jsonl',ledger)]:
 with (ROOT/file).open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({'source_records':len(raw),'canonical_papers':len(canon),'with_abstract':sum(bool(c['abstract']) for c in canon.values()),'without_abstract':sum(not c['abstract'] for c in canon.values()),'merged_source_records':len(raw)-len(canon),'possible_duplicate_records':sum(bool(c['possible_duplicate_ids']) for c in canon.values())},ensure_ascii=False))
