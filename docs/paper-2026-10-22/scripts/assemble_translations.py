"""Assemble directly authored Chinese abstracts; this script never calls a model.

Edit translations/*.json to add prose. The script only validates and packages it.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
papers = [json.loads(line) for line in (ROOT/'papers.jsonl').read_text().splitlines()]
by_id = {paper['id']:paper for paper in papers}
authored = {}
authors = {}
for path in sorted((ROOT/'translations').glob('*.json')):
    batch = json.loads(path.read_text())
    assert not (authored.keys() & batch.keys()), f'Duplicate IDs in {path}'
    authored.update(batch)
    authors.update({paper_id: 'astra_subagent' if path.name.startswith('astra-') else 'current_conversation_assistant' for paper_id in batch})
assert authored.keys() <= by_id.keys(), 'Unknown paper ID'
records = []
review_path = ROOT/'translation-work/main-review.json'
reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
for paper in papers:
    if paper['id'] not in authored:
        continue
    prose = authored[paper['id']]
    assert paper['abstract'], paper['id']
    assert isinstance(prose,str) and len(re.findall(r'[\u4e00-\u9fff]',prose)) > 30, paper['id']
    review = 'directly_authored'
    if authors[paper['id']] == 'astra_subagent':
        checked_hash = reviews.get(paper['id'], {}).get('translation_sha256')
        review = 'main_assistant_checked' if checked_hash == hashlib.sha256(prose.encode()).hexdigest() else 'pending_main_assistant_review'
    records.append({'id':paper['id'], 'source_sha256':hashlib.sha256(paper['abstract'].encode()).hexdigest(),
                    'abstract_zh':prose, 'author':authors[paper['id']],
                    'method':'direct_reading_and_chinese_rewriting',
                    'review':review, 'date':'2026-10-23'})
(ROOT/'abstracts-zh.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
audit = {'date':'2026-10-23', 'source_abstracts':sum(bool(p['abstract']) for p in papers),
         'chinese_abstracts':len(records),
         'remaining':sum(bool(p['abstract']) and p['id'] not in authored for p in papers),
         'missing_source_abstracts':sum(not p['abstract'] for p in papers),
         'by_conference':dict(Counter(by_id[r['id']]['conference'] for r in records)),
         'authors':dict(Counter(authors.values())),
         'review_status':dict(Counter(r['review'] for r in records)),
         'method':'Direct reading and Chinese rewriting by the main assistant and user-authorized Astra subagents; no external translation API used.',
         'coverage_status':'complete' if len(records)==sum(bool(p['abstract']) for p in papers) else 'in_progress',
         'checks':['valid paper IDs','no duplicate translations','source abstracts present','Chinese text present','source SHA-256 recorded']}
(ROOT/'translation-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2))
