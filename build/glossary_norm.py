"""Normalize TechMC Glossary rows into concept records (recognition layer)."""
import csv, json, re, collections

SRC = '/root/dl/glossary/TechMC-Glossary/TechMC Glossary.csv'
rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
corpus = json.load(open('/root/build/corpus.json', encoding='utf-8'))
doc_text = [(d['title'] + ' ' + ' '.join(t for _, t in d['parts'])).lower() for d in corpus['docs']]


def split_zh(s):
    s = s.replace('*', '').strip()
    s = re.sub(r'（[^）]*）|\([^)]*\)', lambda m: '' if m.group(0)[1:-1] in ('实体的', '村民', 'Shulker Box') else m.group(0), s)
    out = []
    for part in re.split(r'[/／;；,，]', s):
        part = part.strip()
        # "即时(瞬时)" -> 即时, 瞬时 ; "临时存储(缓存)" -> both
        m = re.match(r'^(.*?)[（(](.*?)[）)]$', part)
        if m:
            out += [m.group(1).strip(), m.group(2).strip()]
        elif part:
            out.append(part)
    return [x for x in out if x]


def split_en(s):
    s = s.replace('*', '').strip()
    main = re.sub(r'\s*\(aka[^)]*\)?', '', s)
    akas = re.findall(r'aka\s+([^)]*)', s)
    out = []
    for part in [main] + [a for x in akas for a in x.split('/')]:
        for p in re.split(r'/', part):
            p = re.sub(r'\([^)]*\)', '', p).strip(' .')
            if p: out.append(p)
    # "Update Suppression/Suppressor" style already split; also strip parenthetical prefixes "(TNT) Basket"
    return list(dict.fromkeys(out))


concepts = collections.OrderedDict()
for i, r in enumerate(rows):
    zh = r['Chinese'].strip()
    if not zh:
        continue
    key = re.sub(r'[\s*]', '', zh)
    c = concepts.get(key)
    if not c:
        c = concepts[key] = {'id': f'g{i}', 'name': split_zh(zh)[0] if split_zh(zh) else zh, 'zh_raw': zh,
                             'category': set(), 'aliases_zh': [], 'aliases_en': [], 'short': [],
                             'desc': '', 'desc_en': '', 'rows': []}
    c['rows'].append(i)
    c['category'].update(x.strip() for x in r['Category'].split(';'))
    for a in split_zh(zh):
        if a not in c['aliases_zh']: c['aliases_zh'].append(a)
    for a in split_en(r['Full Form (English)']):
        if a not in c['aliases_en']: c['aliases_en'].append(a)
    sf = r['Short Form'].strip()
    if sf and sf not in c['short']: c['short'].append(sf)
    if len(r['Description (Chinese)'].strip()) > len(c['desc']): c['desc'] = r['Description (Chinese)'].strip()
    if len(r['Description'].strip()) > len(c['desc_en']): c['desc_en'] = r['Description'].strip()


def df(alias):
    a = alias.lower()
    return sum(1 for t in doc_text if a in t)


out = []
for c in concepts.values():
    c['category'] = sorted(c['category'])
    c['alias_df'] = {a: df(a) for a in c['aliases_zh'] + c['aliases_en'] + c['short']}
    out.append(c)
json.dump(out, open('/root/build/glossary_norm.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(rows), 'rows ->', len(out), 'concepts')
print('merged duplicates:', [(c['name'], c['rows']) for c in out if len(c['rows']) > 1])
