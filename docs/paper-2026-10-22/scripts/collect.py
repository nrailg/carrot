"""Collect public 2026 conference metadata; run with uv --no-project --with requests --with beautifulsoup4."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, parse_qs, urlparse
import hashlib
import gzip
import json
import re
import time
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'sources'
MANIFEST = []

def read_text(path):
    return (path.read_text() if path.exists()
            else gzip.decompress(Path(str(path)+'.gz').read_bytes()).decode('utf-8'))

def fetch(name, url):
    p = RAW / name
    packed = Path(str(p)+'.gz')
    if not packed.exists() and (not p.exists() or p.stat().st_size < 1000):
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                # PaperCept declares ISO-8859-1 but uses Windows-1252 punctuation.
                r.encoding = 'windows-1252' if 'papercept' in url else 'utf-8'
                p.write_text(r.text, encoding='utf-8')
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2)
    actual = p if p.exists() else packed
    data = p.read_bytes() if p.exists() else gzip.decompress(packed.read_bytes())
    return p, {'file': str(actual.relative_to(ROOT)), 'url': url,
               'retrieved_date': '2026-10-22', 'bytes_saved': len(data),
               'sha256': hashlib.sha256(data).hexdigest()}

def soup(p):
    return BeautifulSoup(read_text(p), 'html.parser')

def clean(t):
    return re.sub(r'\s+', ' ', re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', t)).strip()

def add_source(name, url):
    p, m = fetch(name, url)
    MANIFEST.append(m)
    return p

def record(venue, pid, title, authors, url, **kw):
    return dict(id=f'{venue.lower()}-2026:{pid}', conference=venue, year=2026,
                paper_id=pid, title=clean(title), authors=authors, url=url,
                abstract='', abstract_source='', keywords=[], presentations=[],
                track='main-program', **kw)

def collect_corl():
    url = 'https://2026.corl.org/program/accepted-papers'
    s = soup(add_source('corl-accepted.html', url))
    embed = next(t['data-code'] for t in s.select('[data-code]')
                 if 'openreview.net/forum?id=' in t['data-code'])
    (RAW / 'corl-accepted-embed.html').write_text(embed)
    s = BeautifulSoup(embed, 'html.parser')
    meta = json.loads(read_text(add_source('corl-data.json',
        'https://gisbi-kim.github.io/conference-explorers/corl2026/data.json')))
    enrich = {x['id']: x for x in meta['papers']}
    out = {}
    for a in s.select('a[href*="openreview.net/forum?id="]'):
        pid = parse_qs(urlparse(a['href']).query)['id'][0]
        title = clean(a.get_text(' ', strip=True))
        if pid not in out:
            out[pid] = record('CoRL', pid, title, [], a['href'],
                              source_url=url, source_type='official_accepted_program')
        assert out[pid]['title'] == title, (pid, title)
        parent = a.find_parent(class_='session')
        label = clean(parent.find('h3').get_text(' ', strip=True)) if parent else ''
        if label and label not in out[pid]['presentations']:
            out[pid]['presentations'].append(label)
    for pid, p in out.items():
        x = enrich.get(pid, {})
        p['authors'] = x.get('authors', [])
        p['abstract'] = clean(x.get('abstract', ''))
        p['abstract_source'] = x.get('abstract_source', '')
        p['abstract_retrieval'] = 'community_metadata_snapshot_2026-10-18'
        p['metadata_mirror'] = 'https://gisbi-kim.github.io/conference-explorers/corl2026/data.json'
        p['metadata_title'] = x.get('title', '')
        p['keywords'] = x.get('keywords', [])
        p['pdf_url'] = x.get('pdf', '')
        p['arxiv_url'] = x.get('arxiv', '')
        p['project_urls'] = x.get('project', [])
        p['code_urls'] = x.get('code', [])
        p['presentation_type'] = x.get('presentation_type', '')
    return list(out.values()), {'official_unique': len(out),
        'presentation_events': len(s.select('a[href*="openreview.net/forum?id="]')),
        'mirror_unique': len(enrich), 'official_missing_from_mirror': sorted(set(out)-set(enrich)),
        'mirror_extra_ids': sorted(set(enrich)-set(out)),
        'mirror_title_differences': [p['id'] for p in out.values()
                                   if p['title'] != clean(p['metadata_title'])]}

def collect_rss():
    base = 'https://roboticsproceedings.org/rss22/'
    s = soup(add_source('rss-index.html', base))
    links = [a for a in s.select('a[href]') if re.fullmatch(r'p\d+\.html', a['href'])]
    def get(a):
        href = a['href']; pid = href[:-5]
        path, manifest = fetch('rss-' + href, urljoin(base, href))
        sp = soup(path)
        authors = [a['content'] for a in sp.select('meta[name="citation_author"]')]
        title = sp.select_one('meta[name="citation_title"]')['content']
        p = record('RSS', pid, title, authors,
                   urljoin(base, href), source_url=base, source_type='official_proceedings')
        text = sp.select_one('.content').get_text(' ', strip=True)
        p['abstract'] = clean(text.split('Abstract:', 1)[1].split('Download:', 1)[0])
        p['abstract_source'] = p['url']
        p['abstract_retrieval'] = 'official'
        p['pdf_url'] = urljoin(base, pid + '.pdf')
        p['doi'] = f'10.15607/RSS.2026.XXII.{int(pid[1:]):03}'
        return p, manifest
    with ThreadPoolExecutor(max_workers=6) as ex:
        result = list(ex.map(get, links))
    MANIFEST.extend(m for _, m in result)
    return [p for p, _ in result], {'official_unique': len(result)}

def collect_icra():
    base = 'https://ras.papercept.net/conferences/conferences/ICRA26/program/'
    index = soup(add_source('icra-index.html', base))
    days = list(dict.fromkeys(a['href'] for a in index.select('a[href]')
                             if re.fullmatch(r'ICRA26_ContentListWeb_\d+\.html', a['href'])))
    papers = {}; daily = {}; occurrences = 0
    for href in days:
        n = re.search(r'_(\d+)\.html', href).group(1)
        s = soup(add_source(f'icra-day{n}.html', urljoin(base, href)))
        abstracts = {x['id']: x for x in s.select('div[id^="Ab"]')}
        session_cache = {}
        daily[href] = 0
        for a in s.select('.pTtl a[onclick]'):
            match = re.search(r"viewAbstract\('([^']+)'", a['onclick'])
            if not match: continue
            pid = match.group(1); daily[href] += 1; occurrences += 1
            tr = a.find_parent('tr'); table = tr.find_parent('table')
            header = tr.find_previous_sibling('tr', class_='pHdr')
            anchor = header.find('a', attrs={'name': True}) if header else None
            ab = abstracts.get('Ab' + pid)
            assert ab, pid
            authors = []
            for row in tr.find_next_siblings('tr'):
                if row.select_one('div[id^="Ab"]'): break
                if 'pHdr' in row.get('class', []): break
                authors.extend(x.get_text(' ', strip=True) for x in row.select('a[href*="AuthorIndexWeb"]'))
            keywords = [x.get_text(' ', strip=True) for x in ab.select('a[href*="KeywordIndexWeb"]')]
            text = ab.get_text(' ', strip=True)
            abstract = clean(text.split('Abstract:', 1)[1]) if 'Abstract:' in text else ''
            p = record('ICRA', pid, a.get_text(' ', strip=True), authors,
                urljoin(base, href) + ('#' + anchor['name'] if anchor else '#Ab' + pid),
                source_url=urljoin(base, href), source_type='official_technical_program')
            p['abstract'] = abstract; p['abstract_source'] = p['url']; p['abstract_retrieval'] = 'official'
            p['keywords'] = keywords
            table_key = id(table)
            if table_key not in session_cache:
                session_headers = table.select('tr.sHdr b')
                session_cache[table_key] = ' · '.join(clean(x.get_text(' ', strip=True)) for x in session_headers[:2])
            session = session_cache[table_key]
            p['presentations'] = [session]
            if 'Late Breaking Results' in session:
                p['track'] = 'late-breaking-results'
            p['program_label'] = clean(header.get_text(' ', strip=True)).split('Add to')[0].strip() if header else ''
            if pid in papers:
                assert papers[pid]['title'] == p['title']
                papers[pid]['presentations'].extend(p['presentations'])
            else: papers[pid] = p
    return list(papers.values()), {'official_unique': len(papers), 'paper_events': occurrences, 'daily': daily}

def collect_iros():
    url = 'https://2026.ieee-iros.org/program/paper-index/'
    s = soup(add_source('iros-official.html', url))
    data = json.loads(s.find(id='pi-data').string)
    poster_url = 'https://2026.ieee-iros.org/program/poster-sessions/'
    sp = soup(add_source('iros-posters.html', poster_url))
    full = json.loads(sp.find(id='poster-map-data').string)['papers']
    out = []
    for x in data:
        pid = x['pn']; more = full.get(pid, {})
        p = record('IROS', pid, x['title'], [a.strip() for a in x['authors'].split(';')],
            f'https://2026.ieee-iros.org/program/contributed-talks/#{x["sid"]}:{pid}',
            source_url=url, source_type='official_technical_program')
        p['keywords'] = [k.strip() for k in x['keywords'].split(';') if k.strip()]
        p['abstract'] = clean(more.get('ab', '')); p['abstract_source'] = poster_url + '#pn:' + pid
        p['abstract_retrieval'] = 'official'; p['presentations'] = [x['session']]
        p['presentation_type'] = x['type']; p['day'] = x['day']
        out.append(p)
    # Poster-only LBR abstracts are a separate population from the main paper index.
    for pid in sorted(set(full) - {x['pn'] for x in data}):
        x = full[pid]
        assert x['sn'] == 'Late-Breaking Results', (pid, x['sn'])
        p = record('IROS', pid, x['t'], [a.strip() for a in x['au'].split(';')],
                   poster_url + '#pn:' + pid, source_url=poster_url,
                   source_type='official_lbr_poster_program')
        p['track'] = 'late-breaking-results'
        p['abstract'] = clean(x.get('ab', '')); p['abstract_source'] = p['url']
        p['abstract_retrieval'] = 'official'
        p['keywords'] = [k.strip() for k in x.get('kw', '').split(';') if k.strip()]
        p['presentations'] = [x['sn']]
        out.append(p)
    old_source=add_source('iros-explorer.html', 'https://gisbi-kim.github.io/iros2026-explorer/output/iros2026_explorer.html')
    old = json.loads(soup(old_source).find(id='papers-data').string)
    old_ids = {x['paper_number'] for x in old}; ids = {x['pn'] for x in data}
    return out, {'official_unique': len(data), 'lbr_unique': len(out)-len(data),
        'total_including_lbr': len(out), 'poster_metadata_unique': len(full),
        'missing_poster_metadata': sorted(ids-set(full)), 'poster_extra_ids': sorted(set(full)-ids),
        'old_community_count': len(old), 'old_ids_absent_in_current_index': sorted(old_ids-ids),
        'current_ids_absent_in_old_snapshot': sorted(ids-old_ids),
        'presentation_types': dict(Counter(p.get('presentation_type', 'LBR') for p in out))}

if __name__ == '__main__':
    papers = []; audit = {}
    for venue, fn in [('CoRL', collect_corl), ('RSS', collect_rss), ('ICRA', collect_icra), ('IROS', collect_iros)]:
        records, stats = fn(); papers.extend(records); audit[venue] = stats
        stats['abstracts'] = sum(bool(p['abstract']) for p in records)
        stats['authors'] = sum(bool(p['authors']) for p in records)
        print(venue, json.dumps(stats, ensure_ascii=False), flush=True)
    assert len(papers) == len({p['id'] for p in papers})
    (ROOT/'papers.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in papers))
    (ROOT/'collection-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    (ROOT/'source-manifest.json').write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2)+'\n')
