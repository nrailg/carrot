"""Build the self-contained representative-paper reader from reviewed explanations."""
import json
from pathlib import Path
from batch_io import FIELDS
from review_io import fingerprint

R = Path(__file__).resolve().parents[1]
editorial = json.loads((R / 'map-editorial.json').read_text())
explanations = json.loads((R / 'representative-explanations.json').read_text())
index = {p['paper_id']: p for p in map(json.loads, (R.parent / 'paper-summaries.jsonl').read_text().splitlines())}
papers = []
for category in editorial['categories']:
    for pid in category['representatives']:
        source, text = index[pid], explanations[pid]
        assert text['source_summary_sha256'] == fingerprint({k: source[k] for k in FIELDS}), pid
        title = source['title']
        short = title.split(': ', 1)[0]
        # Use the acronym when present; otherwise retain the full descriptive title.
        short = short if len(short) <= 35 else title
        papers.append({'paper_id': pid, 'category': category['id'], 'order': len(papers) + 1,
                       'title': title, 'short_title': short, 'url': source['evidence_url'],
                       **{k: text[k] for k in ('problem', 'solution', 'limitation')}})
assert len(papers) == 57 and len({p['paper_id'] for p in papers}) == 57
payload = json.dumps({'categories': editorial['categories'], 'papers': papers,
                      'glossary': editorial['glossary'], 'limitations': editorial['limitations']}, ensure_ascii=False)
payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
template = (R / 'templates/problem-map.html').read_text()
assert template.count('__PAPER_DATA__') == 1
out = R.parent / 'problem-map.html'
out.write_text(template.replace('__PAPER_DATA__', payload))
print(f'Wrote {out}: {len(papers)} representatives, {out.stat().st_size:,} bytes')
