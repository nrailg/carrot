"""Evidence-based, deliberately broad AI screening. No external model/API is called."""
from pathlib import Path
from collections import Counter
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

# These tags describe AI methods, not the robot's application domain.
PATTERNS = {
    'vla-foundation': r'vision[ -]language[ -]action|VLA(?:s)?\b|\bVLM(?:s)?\b|vision[ -]language|large[ -](?:language|vision|behavior|behaviour|action|multimodal)[ -]models?|\bLLM(?:s)?\b|foundation(?:al)? models?|\b(?-i:LoRA)\b|\bOpenVLA\b|\bpi[- ]?0(?:\.5)?\b|π[₀0]',
    'reinforcement-learning': r'reinforcement learning|\b(?-i:RL|MARL|DRL)(?:-based)?\b|\bPPO\b|\b(?-i:SAC)\b|actor[ -]critic|policy gradient|reward (?:learning|modeling|modelling)|offline (?:RL|reinforcement)|inverse reinforcement',
    'imitation-policy': r'imitation learning|behavior(?:al)? cloning|behaviour(?:al)? cloning|learning from (?:demonstration|human|video)|demonstration learning|visuomotor polic|diffusion polic|flow polic|action chunk|\bDAgger\b|\b(?-i:ACT)\b(?=.*(?:policy|policies|chunk|transformer))',
    'generative-models': r'diffusion (?:model|polic|transformer)|flow[ -]matching|rectified flow|generative (?:model|polic|AI|world|prior)|denoising diffusion|\bDDPM\b|\bGANs?\b|variational autoencoder|\bVAE\b',
    'world-models': r'world[ -]model|learned (?:world|dynamics|simulator)|neural (?:dynamics|simulat)|video (?:generation|prediction|model)|latent (?:dynamics|imagination)|model-based reinforcement',
    'representation-adaptation': r'representation learning|self[ -]supervised|contrastive learning|transfer learning|continual learning|meta[ -]learning|few[ -]shot learning|pre[ -]?train|fine[ -]?tun|knowledge distillation|policy distillation|domain adaptation|test[ -]time (?:adaptation|training)|in[ -]context learning',
    'learned-perception': r'deep learning for visual perception|visual learning|neural (?:radiance|implicit|field|map|rendering)|\bNeRF\b|open[ -]vocabulary|semantic (?:segmentation|scene understanding)|panoptic segmentation|object detection|point cloud (?:learning|network)|convolutional|\bCNN\b|\bCLIP\b|\bDINO(?:v[23])?\b|\bLPIPS\b|segment anything|vision transformer',
    'neural-control': r'machine learning for robot control|model learning for control|neural (?:network|controller|control|operator|ODE)|graph neural|physics[ -]informed neural|\bPINNs?\b|\bMLPs?\b|multi[ -]layer perceptron|deep (?:neural|learning)|gaussian process(?:es)?|learning[ -]based (?:control|dynamics)|neural and fuzzy control',
    'symbolic-ai-planning': r'\bPOMDPs?\b|\bPDDL(?:Stream)?\b|neuro[ -]symbolic|symbolic (?:planning|reasoning)|monte[ -]carlo tree search|\bMCTS\b|cognitive (?:control architecture|modeling)|belief[ -]space planning|planning under uncertainty|cognitive control architectures',
    'ai-data-evaluation': r'data\s*sets? for robot learning|data sets for robotic vision|robot learning (?:benchmark|dataset)|benchmark.*(?:VLA|foundation model|robot polic)|dataset.*(?:imitation|robot learning)|(?:VLA|foundation model|robot polic).*(?:benchmark|dataset)',
    'other-robot-learning': r'robot learning|sensorimotor learning|bioinspired robot learning|learning from experience|\bAI[ -](?:based|enabled|driven)|artificial intelligence|machine learning|supervised learning|unsupervised learning|learning and adaptive systems|integrated planning and learning|federated learning|active learning|online learning|learning[ -]based|learning (?:to|from) |end[ -]to[ -]end learning',
}
COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
LABELS = {
    'vla-foundation': 'VLA / VLM / LLM / 基础模型',
    'reinforcement-learning': '强化学习与奖励学习',
    'imitation-policy': '模仿学习与动作策略',
    'generative-models': '扩散 / 流匹配 / 生成模型',
    'world-models': '世界模型与学习动力学',
    'representation-adaptation': '表征、迁移与模型适配',
    'learned-perception': '学习式感知与神经场景表示',
    'neural-control': '神经网络与学习式控制',
    'symbolic-ai-planning': '符号 AI、认知与不确定性规划',
    'ai-data-evaluation': 'AI 数据集、评测与基础设施',
    'other-robot-learning': '其他机器学习与机器人学习',
}

def norm(text):
    return unicodedata.normalize('NFKC', text).replace('–', '-').replace('—', '-').replace('‑', '-').replace('−', '-')

def matches(text):
    text = norm(text)
    return {k: list(dict.fromkeys(m.group(0) for m in r.finditer(text)))[:5]
            for k, r in COMPILED.items() if r.search(text)}

def classify(p):
    title = matches(p['title']); keys = matches('; '.join(p['keywords']))
    abstract = matches(p['abstract'])
    evidence = []
    for field, tags in [('title', title), ('keywords', keys), ('abstract', abstract)]:
        for tag, terms in tags.items():
            evidence.append({'field': field, 'tag': tag, 'terms': terms})
    direct = set(title) | set(keys)
    # "Learning" alone is ambiguous (e.g. human education, iterative learning control).
    learning_title = bool(re.search(r'\blearn(?:ing|ed|s)?\b', p['title'], re.I))
    if p['conference'] == 'CoRL':
        status = 'ai-related'
        basis = 'CoRL 官方录用论文；机器人学习会议范围内的研究及配套系统 / 数据 / 评测。'
    elif direct:
        status = 'ai-related'
        basis = '标题或作者关键词明确涉及 AI 方法 / 模型 / AI 数据与评测。'
    elif abstract:
        # Abstract-only hits need context: they may describe baselines or related work.
        status = 'candidate'
        basis = '仅摘要出现 AI 线索；可能是本文方法，也可能是背景 / 对比基线，需复核。'
    elif learning_title:
        status = 'candidate'
        basis = '标题有 learning / learned，尚未从明确方法词确认是否为机器学习。'
    else:
        status = 'unmarked'
        basis = '现有字段未命中明确 AI 证据；不等同于已确认与 AI 无关。'
    method_sentences = []
    if status == 'candidate' and abstract:
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])', p['abstract']):
            if re.search(r'\b(?:we (?:propose|present|introduce|develop|design|train|learn|use|employ|evaluate|investigate|demonstrate)|our (?:method|approach|framework|model|policy|system|network)|(?:proposed|introduced) (?:method|approach|framework|model|policy|system|network))\b', sentence, re.I) and matches(sentence):
                method_sentences.append(sentence)
        if method_sentences:
            status = 'ai-related'
            basis = '摘要描述本文方法 / 系统 / 评测的句子中出现明确 AI 方法词；自动上下文筛选，保留证据句供核对。'
        elif learning_title and set(abstract) - {'other-robot-learning'}:
            status = 'ai-related'
            basis = '标题明确研究 learning / learned，摘要提供具体 AI 模型或方法证据。'
    tags = [t for t in PATTERNS if t in direct]
    abstract_tags = [t for t in PATTERNS if t in abstract and t not in direct]
    p['ai_status'] = status
    p['ai_basis'] = basis
    p['ai_tags'] = tags or (abstract_tags if status != 'unmarked' else [])
    p['ai_abstract_tags'] = abstract_tags
    p['ai_evidence'] = evidence
    p['ai_review'] = 'deterministic_screen_v1'
    p['ai_method_evidence'] = method_sentences[:2]
    return p

if __name__ == '__main__':
    papers = [classify(json.loads(line)) for line in (ROOT/'papers.jsonl').read_text().splitlines()]
    # Human review is a separate, ID-keyed overlay, so it survives regeneration.
    override = ROOT/'review-overrides.json'
    overrides = json.loads(override.read_text()) if override.exists() else {}
    for p in papers:
        if p['id'] in overrides:
            p.update(overrides[p['id']]); p['ai_review'] = 'manual_abstract_review'
    (ROOT/'papers.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in papers))
    for status, name in [('ai-related','ai-papers.jsonl'), ('candidate','ai-candidates.jsonl')]:
        selected = [p for p in papers if p['ai_status'] == status]
        (ROOT/name).write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in selected))
    stats = {}
    for venue in ['CoRL', 'RSS', 'ICRA', 'IROS']:
        group = [p for p in papers if p['conference'] == venue]
        stats[venue] = {'total': len(group), 'abstracts': sum(bool(p['abstract']) for p in group),
            'statuses': dict(Counter(p['ai_status'] for p in group)),
            'tags': dict(Counter(t for p in group if p['ai_status'] == 'ai-related' for t in p['ai_tags']))}
    (ROOT/'screening-stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
