"""Render Markdown catalogs and a self-contained, offline paper browser."""
from pathlib import Path
from collections import Counter
import html
import json
import re
import runpy
import sys
import markdown
sys.path.insert(0, str(Path(__file__).parent))
from classify import LABELS

ROOT = Path(__file__).resolve().parents[1]
PAPERS = [json.loads(l) for l in (ROOT/'papers.jsonl').read_text().splitlines()]
# Translations are separate derived data; never overwrite the source abstracts.
import hashlib
translation_path = ROOT/'abstracts-zh.jsonl'
translations = {r['id']: r for r in map(json.loads, translation_path.read_text().splitlines())} if translation_path.exists() else {}
for paper in PAPERS:
    record = translations.get(paper['id'])
    if record:
        assert record['source_sha256'] == hashlib.sha256(paper['abstract'].encode()).hexdigest(), paper['id']
        paper['abstract_zh'] = record['abstract_zh']
        paper['translation_review'] = record['review']
translated_count = sum(bool(p.get('abstract_zh')) for p in PAPERS)
VENUES = ['CoRL', 'RSS', 'ICRA', 'IROS']
STATUS = {'ai-related':'AI 相关', 'candidate':'AI 候选，待复核', 'unmarked':'未标记'}

def esc(s):
    return str(s).replace('|','\\|').replace('\n',' ')

def tags(p):
    return '；'.join(LABELS[t] for t in p['ai_tags']) or '机器人学习配套研究'

def evidence(p):
    if p['ai_review'] == 'manual_abstract_review':
        return '人工复核；' + p['ai_basis']
    hits = [e for e in p['ai_evidence'] if e['field'] in ['title','keywords']]
    if hits:
        terms = list(dict.fromkeys(t for e in hits for t in e['terms']))[:7]
        return '标题 / 作者关键词：' + ', '.join(terms)
    if p['conference'] == 'CoRL':
        return 'CoRL 机器人学习会议范围；具体方法标签参考摘要'
    terms = list(dict.fromkeys(t for e in p['ai_evidence'] for t in e['terms']))[:5]
    return p['ai_basis'] + (' 线索：' + ', '.join(terms) if terms else '')

def table(group, ai=False):
    if ai:
        out=['| ID | 论文（原文 / 官方展示页） | AI 方向 | 纳入依据 |','| --- | --- | --- | --- |']
        for p in group:
            out.append(f'| {p["paper_id"]} | [{esc(p["title"])}]({p["url"]}) | {esc(tags(p))} | {esc(evidence(p))} |')
    else:
        out=['| ID | 论文（原文 / 官方展示页） | 作者 | AI 筛选 |','| --- | --- | --- | --- |']
        for p in group:
            out.append(f'| {p["paper_id"]} | [{esc(p["title"])}]({p["url"]}) | {esc("; ".join(p["authors"]))} | {STATUS[p["ai_status"]]} |')
    return '\n'.join(out)+'\n'

def catalog(title, group, ai=False):
    out=[f'# {title}', '', '快照日期：2026-10-22（Asia/Shanghai）。筛选方法和来源边界见 [README](README.md) 与 [检索审计](collection-audit.md)。', '']
    for venue in VENUES:
        vg=[p for p in group if p['conference']==venue]
        if not vg: continue
        out.extend([f'## {venue} 2026（{len(vg)} 条）',''])
        for track,label in [('main-program','主目录'),('late-breaking-results','Late-Breaking Results（补充摘要 / 展示，单列）')]:
            tg=[p for p in vg if p['track']==track]
            if tg:out.extend([f'### {label}：{len(tg)} 条','',table(tg,ai),''])
    return '\n'.join(out)

(ROOT/'all-papers.md').write_text(catalog('四场会议完整论文目录',PAPERS))
(ROOT/'ai-papers.md').write_text(catalog('AI 相关论文清单',[p for p in PAPERS if p['ai_status']=='ai-related'],True))
(ROOT/'ai-candidates.md').write_text(catalog('AI 关联候选：证据尚不足以确认',[p for p in PAPERS if p['ai_status']=='candidate'],True))
for venue in VENUES:
    txt=catalog(f'{venue} 2026 完整目录',[p for p in PAPERS if p['conference']==venue])
    txt=txt.replace('(README.md)','(../README.md)').replace('(collection-audit.md)','(../collection-audit.md)')
    (ROOT/'catalog'/f'{venue.lower()}-2026.md').write_text(txt)

summary=[]
for venue in VENUES:
    vg=[p for p in PAPERS if p['conference']==venue]
    main=[p for p in vg if p['track']=='main-program'];lbr=[p for p in vg if p['track']!='main-program']
    summary.append(f'| {venue} 2026 | {len(main)} | {sum(p["ai_status"]=="ai-related" for p in main)} | {sum(p["ai_status"]=="candidate" for p in main)} | {len(lbr)} | {sum(bool(p["abstract"]) for p in vg)}/{len(vg)} |')
counts=Counter(p['ai_status'] for p in PAPERS)
main=[p for p in PAPERS if p['track']=='main-program'];lbr=[p for p in PAPERS if p['track']!='main-program']
readme=f'''# CoRL / RSS / ICRA / IROS 最新一期论文扫描

快照：**2026-10-22，Asia/Shanghai**。四个会议均采用最新公开的 **2026 届**。CoRL 2026 将于 11 月 9—12 日举行，使用已公开的录用与展示程序；RSS、ICRA、IROS 采用已举行届次的公开目录。

已完整提取本次来源快照中的 **{len(main):,} 条主目录论文**，另外单列 **{len(lbr)} 条 Late-Breaking Results**，总计 **{len(PAPERS):,} 条会议记录**。主目录包含会议技术程序中的期刊展示论文，不能用该数量计算 contributed-paper 录用率。按会议分别计数，同一工作跨会议展示时不跨会议合并。

| 会议 | 主目录 | AI 相关（主目录） | AI 候选（主目录） | LBR 补充 | 有摘要 / 全部记录 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(summary)}

含 LBR 的总筛选结果：**AI 相关 {counts['ai-related']:,} 条，AI 候选 {counts['candidate']} 条，未标记 {counts['unmarked']:,} 条**。“未标记”表示没有足够的明确证据，不代表已证明与 AI 无关。

## 阅读入口

- [完整目录：全部论文、作者与链接](all-papers.md)
- [AI 相关论文：逐条方向标签与纳入依据](ai-papers.md)
- [AI 候选：需要进一步核对的条目](ai-candidates.md)
- [AI 阅读笔记与优先阅读建议](ai-notes.md)
- [HTML 总览](index.html) · [CoRL](corl.html) · [RSS](rss.html) · [ICRA](icra.html) · [IROS](iros.html)：四个会议各有独立页面，支持 AI 状态、方向、主目录 / LBR 筛选以及全文元数据搜索。
- 按会议查看：[CoRL](catalog/corl-2026.md) · [RSS](catalog/rss-2026.md) · [ICRA](catalog/icra-2026.md) · [IROS](catalog/iros-2026.md)
- [全部结构化记录](papers.jsonl) · [AI 相关 JSONL](ai-papers.jsonl) · [AI 候选 JSONL](ai-candidates.jsonl)
- [来源与完整性审计](collection-audit.md) · [机器可读审计](collection-audit.json) · [来源快照清单与 SHA-256](source-manifest.json)

## 中文摘要

中文摘要覆盖：**{translated_count:,} / {sum(bool(p["abstract"]) for p in PAPERS):,} 条公开摘要**已完成，按语义分段、统一机器人与 AI 术语。中文摘要默认直接显示；英文原文保留在下方，可展开核对。搜索同时覆盖中英文摘要。原始来源缺失的摘要保持空缺。

中文由本次会话助手和用户授权的子代理根据英文摘要撰写，全部经主助手逐篇核对或直接撰写，保留关键方法、实验范围、数字与结论。中英文对照记录与来源哈希保存在 [abstracts-zh.jsonl](abstracts-zh.jsonl)；覆盖情况见 [translation-audit.json](translation-audit.json)。

## AI 筛选口径

AI 相关包括：VLA / VLM / LLM / 基础模型；强化学习与奖励学习；模仿学习与动作策略；扩散 / 流匹配 / 生成模型；世界模型；学习式感知、神经表示与控制；表征学习和迁移；符号 AI、POMDP 等认知 / 不确定性推理；明确面向这些方法的数据、评测与基础设施。硬件论文若明确使用学习方法或服务于机器人学习评测，也可纳入。

普通 MPC、轨迹数值优化、传统状态估计、机械设计与传感器制造，不能仅因用于机器人就标记为 AI。迭代学习控制、无学习的模型驱动扩散优化等列为边界项；会保留人工判断理由。GP 回归等统计学习方法采用较宽口径。

对全部记录运行可复现规则，检查标题、作者关键词和可得摘要，并保留命中字段及方法词。标题 / 关键词明确命中、或摘要明确描述本文 AI 方法时，列入 AI 相关；仅有背景或基线线索时列候选。CoRL 的研究与配套系统按机器人学习会议范围整体纳入，具体方法标签仍逐篇提取。人工复核记录保存在 [review-overrides.json](review-overrides.json)。这是**全目录元数据与摘要扫描，加人工复核**，没有逐篇精读 PDF；AI 分类仍可能有误报和漏报。

方向为多标签，一个方向计数不代表独立论文数。标题、作者关键词标签与仅摘要出现的附加标签在 JSONL 中分别保留，避免把背景提及直接视为论文核心。

## 来源边界

- **CoRL**：官方网页内嵌程序按 OpenReview ID 去重，780 次展示对应 687 篇。作者、摘要等补充来自 2026-10-18 的社区元数据快照，与官方 687 个 ID 全部匹配；不能将这些摘要说成此次直接从 OpenReview 读取。保留原始摘要出处与镜像来源，6 条标题版本差异以官方程序标题为准。
- **RSS**：官方 RSS XXII proceedings，210 篇；全部取得官方摘要。
- **ICRA**：PaperCept 六日日程，提取有论文标题及摘要容器的技术条目；包括 122 条 LBR。有 137 条无公开摘要，保留空值，不补写摘要。
- **IROS**：当前官方 Paper & Author Index 1,929 篇；官方 poster metadata 额外有 149 条 LBR。当前目录比社区旧快照 1,933 篇少 4 条净值，使用当前官方列表；20 篇主目录无摘要。
- 全部作者列表均已取得。没有将搜索结果中的同名文章、作者自报会议备注、被拒稿件或 workshop 投稿混入主目录；不尝试穷举各独立 workshop 的论文。

## 复现

```sh
uv run --no-project --with requests --with beautifulsoup4 python docs/paper-2026-10-22/scripts/collect.py
python3 docs/paper-2026-10-22/scripts/classify.py
python3 docs/paper-2026-10-22/scripts/assemble_translations.py
uv run --no-project --with markdown python docs/paper-2026-10-22/scripts/render.py
python3 docs/paper-2026-10-22/scripts/write_notes.py
python3 docs/paper-2026-10-22/scripts/validate.py
```

默认复用 `sources/` 中本次快照，避免上游后续变化造成目录漂移。完整论文内容与 PDF 可通过逐条链接继续阅读；本目录保存元数据和公开摘要，未批量下载 PDF。

## 浏览器启动

```sh
python3 -m http.server 8769 --bind 127.0.0.1 --directory /Users/wujunyu/work/carrot/docs/paper-2026-10-22
```

启动后访问 http://127.0.0.1:8769/ 。按 Ctrl+C 停止服务。浏览只需要 Python 3，不需要安装前端依赖。
'''
(ROOT/'README.md').write_text(readme)

# No external JavaScript, analytics, network fetches, or dependencies.
payload=json.dumps(PAPERS,ensure_ascii=False).replace('<','\\u003c')
template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robotics 2026 · Paper Scan</title>
<style>body{font:15px/1.65 system-ui,sans-serif;color:#183029;background:#f4f6f3;margin:0}main{max-width:1140px;margin:auto;padding:32px 22px}h1{font-size:30px;letter-spacing:-1px;margin-bottom:4px}p{color:#586b63}header small{letter-spacing:2px;color:#547462}nav{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}input,select,button{font:inherit;padding:9px 12px;border:1px solid #b8c8bd;border-radius:6px;background:white;color:#183029}input{flex:1;min-width:200px}button{cursor:pointer}button:disabled{opacity:.4}article{background:white;border:1px solid #d9e2db;border-radius:9px;padding:18px 22px;margin:12px 0}h2{font-size:18px;margin:4px 0}a{color:#1a6547;text-decoration:none}a:hover{text-decoration:underline}.meta{font-size:13px;color:#6a7a71}.tags{color:#386650;font-size:13px}.basis{font-size:13px;margin:8px 0}.status{display:inline-block;background:#e5efe7;padding:2px 7px;border-radius:4px;font-size:12px}.candidate{background:#fff0cf}summary{cursor:pointer;color:#426c56}.abstract{color:#263e33;max-width:88ch;font-size:16px;line-height:1.9;overflow-wrap:anywhere}.abstract p{color:inherit;margin:0 0 .85em}.abstract-label{font-size:13px;color:#547462;margin:17px 0 9px;font-weight:600}.original{white-space:pre-wrap;font-size:14px;line-height:1.7;margin-top:12px}.translation-note{font-weight:400;color:#7b887f;font-size:12px;margin-left:8px}.pager{display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap;margin:22px 0}.page-jump{display:flex;gap:8px;align-items:center}.page-jump input{flex:none;min-width:0;width:5em;text-align:center}.page-jump label{white-space:nowrap}#result{font-weight:600}footer{margin-top:40px;color:#6a7a71}details{margin-top:10px}.empty{text-align:center;padding:50px}</style>
<main><header><small>RESEARCH INDEX · 2026-10-22</small><h1>CoRL / RSS / ICRA / IROS</h1><p>四场会议全部公开主目录与 LBR 补充。AI 分类基于元数据、摘要规则及人工复核；详情见 <a href="README.md">README</a>。CoRL 摘要为社区元数据快照，其他会议摘要来自官方。</p></header>
<nav><input id="q" type="search" placeholder="搜索标题、作者、关键词、摘要" aria-label="搜索论文"><select id="venue" aria-label="会议"><option value="">全部会议</option><option>CoRL</option><option>RSS</option><option>ICRA</option><option>IROS</option></select><select id="track" aria-label="目录范围"><option value="">主目录 + LBR</option><option value="main-program">主目录</option><option value="late-breaking-results">LBR 补充</option></select><select id="status" aria-label="AI 筛选状态"><option value="">全部 AI 状态</option><option value="ai-related">AI 相关</option><option value="candidate">AI 候选</option><option value="unmarked">未标记</option></select><select id="tag" aria-label="AI 方向"><option value="">全部 AI 方向</option></select></nav>
<div id="result" role="status" aria-live="polite"></div><div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><form class="page-jump" id="jump"><label for="jump-page">跳至</label><input id="jump-page" type="number" min="1" step="1" inputmode="numeric" required aria-label="跳转页码（顶部）"><span>页</span><button type="submit" id="jump-button">跳转</button></form></div><section id="list"></section><div class="pager"><button id="prev2">上一页</button><span id="page2"></span><button id="next2">下一页</button><form class="page-jump" id="jump2"><label for="jump-page2">跳至</label><input id="jump-page2" type="number" min="1" step="1" inputmode="numeric" required aria-label="跳转页码（底部）"><span>页</span><button type="submit" id="jump-button2">跳转</button></form></div><footer>来源保留到逐条记录；标签不代表论文质量或排序。无公开摘要的记录保持空缺。中文摘要已由主助手逐篇核对或直接撰写，英文原文可展开核对。本站离线运行，无外部资源。</footer></main>
<script id="papers" type="application/json">__DATA__</script><script>
const papers=JSON.parse(document.getElementById('papers').textContent),labels=__LABELS__,statusLabels={'ai-related':'AI 相关',candidate:'AI 候选，待复核',unmarked:'未标记'};let page=0,filtered=papers;const size=50,$=id=>document.getElementById(id),escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const [key,value] of Object.entries(labels)){let o=document.createElement('option');o.value=key;o.textContent=value;$('tag').appendChild(o)}
const haystacks=papers.map(p=>[p.title,p.authors.join(' '),p.keywords.join(' '),p.abstract,p.abstract_zh||''].join(' ').toLocaleLowerCase());
function update(){const query=$('q').value.trim().toLocaleLowerCase().split(/\\s+/).filter(Boolean);filtered=papers.filter((p,i)=>(!$('venue').value||p.conference===$('venue').value)&&(!$('track').value||p.track===$('track').value)&&(!$('status').value||p.ai_status===$('status').value)&&(!$('tag').value||p.ai_tags.includes($('tag').value))&&query.every(w=>haystacks[i].includes(w)));page=0;render()}
function abstractBlock(p){
  if(!p.abstract)return '<p class="meta">来源未提供公开摘要。</p>';
  const text=p.abstract_zh||p.abstract;
  const paragraphs=text.split(/\\n\\s*\\n/).filter(Boolean).map(t=>`<p>${escape(t)}</p>`).join('');
  return `<div class="abstract-label">${p.abstract_zh?'中文摘要':'摘要'}<span class="translation-note">${p.abstract_zh?('根据原摘要撰写'):'英文原文 · 中文待补'}</span></div><div class="abstract" lang="${p.abstract_zh?'zh-CN':'en'}">${paragraphs}</div>${p.abstract_zh?`<details class="original-details"><summary>查看英文原文</summary><div class="abstract original" lang="en">${escape(p.abstract)}</div></details>`:''}`;
}
function render(){const pages=Math.max(1,Math.ceil(filtered.length/size));$('result').textContent=`${filtered.length.toLocaleString()} / ${papers.length.toLocaleString()} 条记录`;$('page').textContent=$('page2').textContent=`${page+1} / ${pages}`;for(const suffix of ['', '2']){const input=$('jump-page'+suffix);input.max=pages;input.value=page+1;input.disabled=pages===1;$('jump-button'+suffix).disabled=pages===1}for(const id of ['prev','prev2'])$(id).disabled=page===0;for(const id of ['next','next2'])$(id).disabled=page>=pages-1;$('list').innerHTML=filtered.slice(page*size,(page+1)*size).map(p=>`<article><div class="meta">${escape(p.conference)} 2026 · ${escape(p.paper_id)} · ${p.track==='main-program'?'主目录':'LBR 补充'} <span class="status ${p.ai_status==='candidate'?'candidate':''}">${statusLabels[p.ai_status]}</span></div><h2><a href="${escape(p.url)}" target="_blank" rel="noopener">${escape(p.title)}</a></h2><div class="meta">${escape(p.authors.join('; '))}</div><div class="tags">${escape(p.ai_tags.map(t=>labels[t]).join(' · '))}</div><p class="basis">${escape(p.ai_basis)}</p>${abstractBlock(p)}<details><summary>关键词与筛选证据</summary><p class="meta">关键词：${escape(p.keywords.join('; ')||'未提供')}<br>摘要来源：${escape(p.abstract_retrieval||'未提供')}<br>AI 证据：${escape(p.ai_evidence.map(e=>e.field+': '+e.terms.join(', ')).join(' | '))}<br>复核：${escape(p.ai_review)}</p><a href="${escape(p.source_url)}" target="_blank" rel="noopener">官方目录来源 ↗</a></details></article>`).join('')||'<p class="empty">没有匹配的论文。请缩小关键词范围。</p>'}
for(const suffix of ['', '2'])$('jump'+suffix).addEventListener('submit',event=>{event.preventDefault();const target=Number($('jump-page'+suffix).value),pages=Math.max(1,Math.ceil(filtered.length/size));if(!Number.isInteger(target)||target<1||target>pages)return;page=target-1;render();scrollTo(0,0)});
for(const id of ['venue','track','status','tag'])$(id).addEventListener('change',update);let timer;$('q').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(update,180)});for(const id of ['prev','prev2'])$(id).onclick=()=>{page--;render();scrollTo(0,0)};for(const id of ['next','next2'])$(id).onclick=()=>{page++;render();scrollTo(0,0)};render();
</script></html>'''
# Shared navigation: conference pages contain only that conference's records.
nav='<div class="site-nav"><a href="index.html">总览</a>' + ''.join(
    f'<a href="{v.lower()}.html">{v} 2026</a>' for v in VENUES) + '<a href="ai-notes.html">AI 阅读笔记</a><a href="about.html">来源与说明</a></div>'
extra_css='''.site-nav{display:flex;gap:9px;flex-wrap:wrap;margin:0 0 30px}.site-nav a{padding:6px 12px;border:1px solid #cfdbd2;border-radius:99px;background:white}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:28px 0}.cards article{margin:0}.number{font-size:34px;line-height:1.2;color:#1a6547;font-weight:650}.actions{display:flex;gap:16px;flex-wrap:wrap;margin-top:16px}.doc{background:white;padding:28px;border:1px solid #d9e2db;border-radius:9px}.doc h2{font-size:22px;margin-top:32px}.doc h3{font-size:18px;margin-top:25px}.doc table{border-collapse:collapse;display:block;overflow:auto;width:100%}.doc th,.doc td{border:1px solid #d9e2db;padding:8px 12px;text-align:left}.doc code{font-family:ui-monospace,monospace}.doc pre{overflow:auto;background:#eff4ef;padding:16px;border-radius:6px}.doc p{color:#364c41}@media(max-width:680px){.cards{grid-template-columns:1fr}.doc{padding:18px}}'''
template=template.replace('</style>',extra_css+'</style>').replace('<main><header>','<main>'+nav+'<header>').replace('href="README.md"','href="about.html"')
# Restore filters from a copied URL, including links from the overview cards.
template=template.replace('};render();', '''};
const params=new URLSearchParams(location.search);
for(const id of ['q','track','status','tag']){if(params.has(id))$(id).value=params.get(id)}
update();''')
for venue in [None]+VENUES:
    group=PAPERS if venue is None else [p for p in PAPERS if p['conference']==venue]
    page=template
    ready=sum(bool(p.get('abstract_zh')) for p in group)
    available=sum(bool(p['abstract']) for p in group)
    page=page.replace('</header>',f'<p class="translation-progress">中文摘要：{ready:,} / {available:,} 篇。全部经主助手逐篇核对或直接撰写。缺少公开原文的条目保持空缺。</p></header>')
    if venue:
        page=page.replace('<title>Robotics 2026 · Paper Scan</title>',f'<title>{venue} 2026 · Papers</title>')
        page=page.replace('<h1>CoRL / RSS / ICRA / IROS</h1>',f'<h1>{venue} 2026 · 论文目录</h1>')
        page=page.replace('四场会议全部公开主目录与 LBR 补充。',f'{venue} 2026 的全部公开主目录与 LBR 补充。')
        page=re.sub(r'<select id="venue".*?</select>',f'<input type="hidden" id="venue" value="{venue}">',page)
    data=json.dumps(group,ensure_ascii=False).replace('<','\\u003c')
    (ROOT/(f'{venue.lower()}.html' if venue else 'all.html')).write_text(page.replace('__DATA__',data).replace('__LABELS__',json.dumps(LABELS,ensure_ascii=False)))

style=re.search(r'<style>(.*?)</style>',template,re.S).group(1)
def document(title,body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+style+'</style><main>'+nav+body+'</main></html>'

cards=[]
for venue in VENUES:
    group=[p for p in PAPERS if p['conference']==venue]
    primary=[p for p in group if p['track']=='main-program']
    ai=sum(p['ai_status']=='ai-related' for p in primary)
    supplemental=len(group)-len(primary)
    slug=venue.lower()+'.html'
    cards.append(f'<article><div class="meta">2026 CONFERENCE</div><h2><a href="{slug}">{venue} 2026 ↗</a></h2><div class="number">{len(primary):,}</div><p>主目录论文 · AI 相关 {ai:,} 篇'+(f' · LBR 补充 {supplemental} 条' if supplemental else '')+f'</p><div class="actions"><a href="{slug}">全部论文 →</a><a href="{slug}?status=ai-related">只看 AI →</a><a href="{slug}?status=candidate">待复核 →</a></div></article>')
home='<header><small>ROBOTICS RESEARCH · 2026-10-22</small><h1>机器人顶会 · 论文扫描</h1><p>四场会议分开阅读。共 5,602 篇主目录论文，另有 271 条 LBR。5,716 篇公开摘要已全部完成中文改写及主助手复核，默认直接显示；157 条无公开摘要的记录保留缺失标记。支持中英文搜索与 AI 方向筛选。</p></header><section class="cards">'+''.join(cards)+'</section><article><h2>从 AI 研究问题开始</h2><p>跨机器人迁移、真机在线学习、执行反馈、失败恢复与世界模型：16 篇中文阅读笔记。</p><div class="actions"><a href="ai-notes.html">阅读笔记 →</a><a href="all.html?status=ai-related">跨会议 AI 检索 →</a><a href="all.html">跨会议全部检索 →</a></div></article><p>AI 分类基于标题、关键词、公开摘要与人工复核；CoRL 摘要使用已注明来源的社区快照。<a href="about.html">查看范围与筛选口径</a> · <a href="collection-audit.html">来源审计</a></p>'
(ROOT/'index.html').write_text(document('机器人顶会 · 2026 论文扫描',home))

# Regenerate the authored notes before converting documents, so one build is sufficient.
runpy.run_path(str(ROOT/'scripts/write_notes.py'),run_name='__main__')
link_map={'README.md':'about.html','all-papers.md':'all.html','ai-papers.md':'all.html?status=ai-related','ai-candidates.md':'all.html?status=candidate','ai-notes.md':'ai-notes.html','collection-audit.md':'collection-audit.html',**{f'catalog/{v.lower()}-2026.md':f'{v.lower()}.html' for v in VENUES}}
for source,target,title in [('README.md','about.html','来源与筛选说明'),('ai-notes.md','ai-notes.html','AI 阅读笔记'),('collection-audit.md','collection-audit.html','来源与完整性审计')]:
    body=markdown.markdown((ROOT/source).read_text(),extensions=['tables','fenced_code','toc'])
    body=re.sub(r'href="([^"]+)"',lambda m:'href="'+html.escape(link_map.get(html.unescape(m.group(1)),html.unescape(m.group(1))),quote=True)+'"',body)
    (ROOT/target).write_text(document(title,'<section class="doc">'+body+'</section>'))
print(json.dumps({'records':len(PAPERS),'main':len(main),'lbr':len(lbr),'screening':dict(counts)},ensure_ascii=False))
