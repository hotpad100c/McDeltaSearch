import json, re, sys
sys.path.insert(0, '/root/build')
from graph_spec import M, LINKS, GENERIC, GENERIC_EN
norm = json.load(open('/root/build/glossary_norm.json', encoding='utf-8'))
corpus = json.load(open('/root/build/corpus.json', encoding='utf-8'))
texts = [(d['title'] + ' ' + ' '.join(t for _, t in d['parts'])) for d in corpus['docs']]
low = [t.lower() for t in texts]
classes = set(c for d in corpus['docs'] for c in d['classes'])
is_en = lambda a: bool(re.fullmatch(r'[\x00-\x7f]+', a))

def df(a):
    if is_en(a):
        r = re.compile(r'(?<![A-Za-z0-9])' + re.escape(a.lower()) + r'(?![A-Za-z0-9])')
        return sum(1 for t in low if r.search(t))
    return sum(1 for t in low if a.lower() in t)

problems = []
for mid, (name, terms, cls) in M.items():
    missing = [c for c in cls if c not in classes]
    if missing: problems.append(f'mechanism {mid}: classes not in reports: {missing}')

used_keys = set(); out = []
for c in norm:
    zh = [a for a in c['aliases_zh'] if len(a) >= 2 and a not in GENERIC and '{' not in a]
    en = [a for a in c['aliases_en'] + c['short'] if len(a) >= 2 and a not in GENERIC_EN and '{' not in a]
    link = None
    for k in LINKS:
        if k in c['aliases_zh'] or k == c['name']:
            link = LINKS[k]; used_keys.add(k); break
    extra = link[2] if link else []
    aliases = list(dict.fromkeys(zh + en + extra))
    if not aliases: continue
    dfs = {a: df(a) for a in aliases}
    # English aliases that are common words in the reports are dropped unless explicitly added
    aliases = [a for a in aliases if not (is_en(a) and dfs[a] >= 10 and a not in extra)]
    if not aliases: continue
    role = 'linked' if link else ('jargon' if c['desc'] and max(dfs[a] for a in aliases) < 3 else 'term')
    out.append({'id': c['id'], 'name': c['name'], 'category': c['category'], 'desc': c['desc'],
                'aliases': aliases, 'role': role, 'mechanisms': link[0] if link else [],
                'confidence': link[1] if link else None, 'alias_df': {a: dfs[a] for a in aliases}})
for k in LINKS:
    if k not in used_keys: problems.append(f'link key not found in glossary: {k}')
mech = {k: {'name': v[0], 'terms': v[1], 'classes': v[2]} for k, v in M.items()}
json.dump({'mechanisms': mech, 'concepts': out}, open('/root/build/concepts_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import collections
print(len(out), 'concepts usable;', collections.Counter(c['role'] for c in out))
print('confidence of links:', collections.Counter(c['confidence'] for c in out if c['confidence']))
print('dropped (no usable alias):', len(norm) - len(out))
print('\n'.join(problems))
