"""Validate corpus integrity against saved source manifests and rendered catalogs."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import html as html_lib
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
def read(p):
    return gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
papers=[json.loads(l) for l in (ROOT/'papers.jsonl').read_text().splitlines()]
ids={p['id'] for p in papers}
assert len(papers)==len(ids)==5873
assert all(p['title'] and p['authors'] and all(p['authors']) for p in papers)
assert all(p['url'].startswith('https://') and p['source_url'].startswith('https://') for p in papers)
assert not any(re.search(r'â[\x80-\x9f]|Ã[\x80-\x9f]',p['title']) for p in papers)
counts=Counter((p['conference'],p['track']) for p in papers)
assert counts=={('CoRL','main-program'):687,('RSS','main-program'):210,
               ('ICRA','main-program'):2776,('ICRA','late-breaking-results'):122,
               ('IROS','main-program'):1929,('IROS','late-breaking-results'):149},counts
def source(name):
    p=ROOT/'sources'/name
    return read(p if p.exists() else Path(str(p)+'.gz')).decode('utf-8')
corl_source=html_lib.unescape(source('corl-accepted.html'))
corl_ids=set(re.findall(r'https://openreview\.net/forum\?id=([A-Za-z0-9]+)',corl_source))
assert corl_ids=={p['paper_id'] for p in papers if p['conference']=='CoRL'}
rss_ids=set(re.findall(r'href="(p\d+)\.html"',source('rss-index.html')))
assert rss_ids=={p['paper_id'] for p in papers if p['conference']=='RSS'}
icra_ids=set()
for i in range(1,7):
    icra_ids.update(re.findall(r"viewAbstract\('([^']+)'",source(f'icra-day{i}.html')))
assert icra_ids=={p['paper_id'] for p in papers if p['conference']=='ICRA'}
iros_source=source('iros-official.html')
pi_data=json.loads(re.search(r'<script[^>]+id="pi-data"[^>]*>(.*?)</script>',iros_source,re.S).group(1))
assert {x['pn'] for x in pi_data}=={p['paper_id'] for p in papers if p['conference']=='IROS' and p['track']=='main-program'}
def canonical(s):
    return re.sub(r'[^\w]','',unicodedata.normalize('NFKC',s).casefold())
title_counts=Counter((p['conference'],canonical(p['title'])) for p in papers)
duplicate_titles={f'{v}: {t}':n for (v,t),n in title_counts.items() if n>1}
assert not duplicate_titles,duplicate_titles
for state,name in [('ai-related','ai-papers.jsonl'),('candidate','ai-candidates.jsonl')]:
    subset=[json.loads(l) for l in (ROOT/name).read_text().splitlines()]
    assert {p['id'] for p in subset}=={p['id'] for p in papers if p['ai_status']==state}
overrides=json.loads((ROOT/'review-overrides.json').read_text())
assert set(overrides)<=ids
for p in papers:
    assert p['ai_status'] in ['ai-related','candidate','unmarked']
    if p['ai_status']=='ai-related':
        assert p['conference']=='CoRL' or p['ai_evidence'] or p['ai_review']=='manual_abstract_review',p['id']
for m in json.loads((ROOT/'source-manifest.json').read_text()):
    data=read(ROOT/m['file'])
    assert hashlib.sha256(data).hexdigest()==m['sha256'],m['file']
    assert len(data)==m['bytes_saved'],m['file']
all_md=(ROOT/'all-papers.md').read_text()
rows=[l for l in all_md.splitlines() if re.match(r'^\| (?!ID |---)',l)]
assert len(rows)==len(papers)
for venue in ['CoRL','RSS','ICRA','IROS']:
    md=(ROOT/'catalog'/f'{venue.lower()}-2026.md').read_text()
    assert len([l for l in md.splitlines() if re.match(r'^\| (?!ID |---)',l)])==sum(p['conference']==venue for p in papers)
html=(ROOT/'all.html').read_text()
embedded=json.loads(re.search(r'<script id="papers" type="application/json">(.*?)</script>',html,re.S).group(1))
assert {p['id'] for p in embedded}==ids
translations=[json.loads(l) for l in (ROOT/'abstracts-zh.jsonl').read_text().splitlines()]
translated={p['id']:p for p in translations}
assert len(translated)==len(translations)
assert translated.keys()<=ids
for p in embedded:
    if p['id'] in translated:
        t=translated[p['id']]
        assert t['source_sha256']==hashlib.sha256(p['abstract'].encode()).hexdigest(),p['id']
        assert p['abstract_zh']==t['abstract_zh'],p['id']
        assert t['method']=='direct_reading_and_chinese_rewriting'
    else:
        assert not p.get('abstract_zh'),p['id']
translation_audit=json.loads((ROOT/'translation-audit.json').read_text())
assert translation_audit['chinese_abstracts']==len(translated)
assert translation_audit['remaining']==sum(bool(p['abstract']) for p in papers)-len(translated)
node=shutil.which('node')
assert node,'Node.js required for viewer JavaScript syntax check'
script=re.search(r'</script><script>(.*?)</script>',html,re.S).group(1)
with tempfile.NamedTemporaryFile('w',suffix='.js') as f:
    f.write(script);f.flush();subprocess.run([node,'--check',f.name],check=True,capture_output=True)
conference_pages={}
for venue in ['CoRL','RSS','ICRA','IROS']:
    page=(ROOT/f'{venue.lower()}.html').read_text()
    data=json.loads(re.search(r'<script id="papers" type="application/json">(.*?)</script>',page,re.S).group(1))
    expected={p['id'] for p in papers if p['conference']==venue}
    assert {p['id'] for p in data}==expected,venue
    assert all(p['conference']==venue for p in data),venue
    for p in data:
        assert p.get('abstract_zh')==translated.get(p['id'],{}).get('abstract_zh'),p['id']
    assert '${abstractBlock(p)}<details>' in page
    assert '查看英文原文' in page and '英文原文 · 中文待补' in page
    assert f'<input type="hidden" id="venue" value="{venue}">' in page
    conference_pages[venue]=len(data)
    js=re.search(r'</script><script>(.*?)</script>',page,re.S).group(1)
    with tempfile.NamedTemporaryFile('w',suffix='.js') as f:
        f.write(js);f.flush();subprocess.run([node,'--check',f.name],check=True,capture_output=True)
home=(ROOT/'index.html').read_text()
assert 'id="papers"' not in home
assert all(f'href="{v}.html"' in home for v in ['corl','rss','icra','iros'])
assert (ROOT/'ai-notes.html').exists() and (ROOT/'about.html').exists()
# Browser navigation points to HTML pages and local artifacts.
for p in ROOT.glob('*.html'):
    markup=re.sub(r'<script\b[^>]*>.*?</script>','',p.read_text(),flags=re.S)
    for href in re.findall(r'href="([^"]+)"',markup):
        href=html_lib.unescape(href)
        if '://' in href or href.startswith('#'):continue
        target=href.split('?',1)[0].split('#',1)[0]
        assert (p.parent/target).exists(),(p.name,href)
        assert not target.endswith('.md'),(p.name,href)
# Local Markdown links point to actual delivered artifacts.
for p in ROOT.glob('*.md'):
    for href in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' not in href and not href.startswith('#'):
            if href=='validation.json':
                continue  # This report is created after all checks pass below.
            assert (p.parent/href.split('#')[0]).exists(),(p.name,href)
report={'date':'2026-10-23','result':'PASS','records':len(papers),'unique_ids':len(ids),
        'main_program':sum(p['track']=='main-program' for p in papers),
        'late_breaking_results':sum(p['track']=='late-breaking-results' for p in papers),
        'abstracts':sum(bool(p['abstract']) for p in papers),
        'authors':sum(bool(p['authors']) for p in papers),
        'statuses':dict(Counter(p['ai_status'] for p in papers)),
        'manual_review_overrides':len(overrides),'normalized_duplicate_titles':duplicate_titles,
        'source_checksums':'PASS','official_roster_id_sets':'PASS','markdown_row_counts':'PASS','local_links':'PASS',
        'viewer_embedded_records':'PASS','viewer_javascript_syntax':'PASS',
        'conference_pages':conference_pages,'html_navigation':'PASS',
        'chinese_abstracts':len(translated),'translation_hashes':'PASS',
        'translation_remaining':translation_audit['remaining']}
(ROOT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert (ROOT/'validation.json').exists()
print(json.dumps(report,ensure_ascii=False,indent=2))
