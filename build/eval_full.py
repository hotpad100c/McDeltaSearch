import sys, json, math, re, collections, numpy as np
sys.path.insert(0, '/root/build')
from search_eval import bm25, docs, N, INSTR, W as SW
from model import forward
from emb import tok as stok

G = json.load(open(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.json') else '/root/build/concepts_full.json', encoding='utf-8'))
MECH, CON = G['mechanisms'], G['concepts']
owner = np.load('/root/build/owner.npy'); vs = np.load('/root/build/vecs.npy')
class_df = collections.Counter(c for d in docs for c in set(d['classes']))
is_en = lambda a: bool(re.fullmatch(r'[\x00-\x7f]+', a))
PAT = []
for c in CON:
    for a in c['aliases']:
        if is_en(a):
            PAT.append((re.compile(r'(?<![A-Za-z0-9])' + re.escape(a) + r'(?![A-Za-z0-9])', re.I), c))
        else:
            PAT.append((re.compile(re.escape(a), re.I), c))


def detect_spans(q):
    spans = [(m.start(), m.end(), c) for p, c in PAT for m in p.finditer(q)]
    return [s for s in spans if not any(o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0]) for o in spans)]


def detect(q):
    out = []
    for *_, c in sorted(detect_spans(q), key=lambda s: s[0]):
        if c not in out: out.append(c)
    return out


def strip_aliases(q):
    """Remove recognized jargon from the keyword query so its characters don't match unrelated words."""
    spans = [s for s in detect_spans(q) if s[2]['mechanisms'] or s[2]['role'] == 'jargon']
    keep = [ch for i, ch in enumerate(q) if not any(a <= i < b for a, b, _ in spans)]
    return ''.join(keep)


def enc(text):
    stok.no_padding(); stok.no_truncation()
    ids = stok.encode(INSTR + text).ids[:128]
    return forward(SW, np.array([ids]), np.ones((1, len(ids)), np.float32))[0]


CORPUS = json.load(open('/root/build/corpus.json', encoding='utf-8'))
KEY2IX = {CORPUS['versions'][d['v']]['version'] + '#' + d['slug']: i for i, d in enumerate(CORPUS['docs'])}


def ix(k):
    """Test sets refer to sections by the stable key version#slug, never by list position."""
    if k not in KEY2IX:
        raise KeyError(f'test set refers to a section that no longer exists: {k}')
    return KEY2IX[k]


DCACHE = {}
STRIP = True
def dense(text):
    if text not in DCACHE:
        s = vs @ enc(text); b = np.full(N, -1.0); np.maximum.at(b, owner, s); DCACHE[text] = b
    return DCACHE[text]


def mechs_of(cs):
    return list(dict.fromkeys(m for c in cs for m in c['mechanisms']))


def class_score(mids):
    s = np.zeros(N)
    for mid in mids:
        cls = set(MECH[mid]['classes'])
        for i, d in enumerate(docs):
            hit = cls & set(d['classes'])
            if hit: s[i] += max(math.log(1 + N / class_df[h]) for h in hit)
    return s


def rrf(lists):
    tot = np.zeros(N)
    for s in lists:
        order = np.argsort(-s)[:150]
        for r, i in enumerate(order):
            if s[i] > 0 or s.min() < 0: tot[i] += 1 / (60 + r)
    return tot


def run(q, use_graph=True):
    cs = detect(q) if use_graph else []
    kq = strip_aliases(q) if (use_graph and STRIP) else q
    lists = [bm25(kq), dense(q)]
    mids = mechs_of(cs)
    if mids:
        lists.append(bm25(' '.join(t for m in mids for t in MECH[m]['terms'])))
        lists.append(class_score(mids))
    desc = ' '.join(c['desc'] for c in cs if c['role'] == 'jargon' and not c['mechanisms'])
    if desc:
        lists.append(bm25(desc))
    return rrf(lists), cs


def evaluate(qs, use_graph):
    h10 = rec = mrr = 0
    for q, rel in qs:
        s, _ = run(q, use_graph)
        top = list(np.argsort(-s)[:10]); hits = [i for i in top if i in rel]
        h10 += bool(hits); rec += len(hits) / min(len(rel), 10)
        first = next((r for r, i in enumerate(top) if i in rel), None)
        mrr += 0 if first is None else 1 / (first + 1)
    n = len(qs); return h10 / n, rec / n, mrr / n


if __name__ == '__main__':
    J = json.load(open('/root/build/eval_jargon.json', encoding='utf-8'))
    jq = [(q, {ix(k2) for k2 in J['sets'][k]}) for q, k in J['questions']]
    oq = [(q, {ix(t)}) for t, q in json.load(open('/root/build/eval_questions.json', encoding='utf-8'))]
    print('jargon questions, detected concepts:')
    for q, _ in jq: print('  ', q, '->', [c['name'] for c in detect(q)])
    print('original questions that now trigger concepts:')
    for q, _ in oq:
        cs = detect(q)
        if cs: print('  ', q, '->', [(c['name'], c['role']) for c in cs])
    for label, qs in [('jargon (15)', jq), ('original (51)', oq)]:
        a, b = evaluate(qs, False), evaluate(qs, True)
        print(f'{label:14s} no graph: hit@10 {a[0]:.2f} recall {a[1]:.2f} MRR {a[2]:.3f} | full glossary graph: hit@10 {b[0]:.2f} recall {b[1]:.2f} MRR {b[2]:.3f}')
