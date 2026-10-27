import sys, json, math, collections, numpy as np
sys.path.insert(0, '/root/build')
from search_eval import bm25, docs, N, INSTR, W as SW, terms
from model import forward
from emb import tok as stok

G = json.load(open('/root/build/concepts.json', encoding='utf-8'))['concepts']
J = json.load(open('/root/build/eval_jargon.json', encoding='utf-8'))
owner = np.load('/root/build/owner.npy'); vs = np.load('/root/build/vecs.npy')

class_df = collections.Counter(c for d in docs for c in set(d['classes']))


def detect(q):
    ql = q.lower(); spans = []
    for c in G:
        for a in c['aliases']:
            k = ql.find(a.lower())
            while k >= 0:
                spans.append((k, k + len(a), c)); k = ql.find(a.lower(), k + 1)
    keep = [s for s in spans if not any(o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0]) for o in spans)]
    out = []
    for _, _, c in keep:
        if c not in out: out.append(c)
    return out


def enc(text):
    stok.no_padding(); stok.no_truncation()
    ids = stok.encode(INSTR + text).ids[:128]
    return forward(SW, np.array([ids]), np.ones((1, len(ids)), np.float32))[0]


def dense(text):
    s = vs @ enc(text); b = np.full(N, -1.0); np.maximum.at(b, owner, s); return b


def class_score(concepts):
    s = np.zeros(N)
    for c in concepts:
        for m in c['mechanisms']:
            cls = set(m['classes'])
            for i, d in enumerate(docs):
                hit = cls & set(d['classes'])
                if hit:
                    s[i] += max(math.log(1 + N / class_df[h]) for h in hit)
    return s


def expanded_terms(concepts):
    return ' '.join(t for c in concepts for m in c['mechanisms'] for t in m['terms'])


def rrf(lists):
    tot = np.zeros(N)
    for s in lists:
        order = np.argsort(-s)[:150]
        for r, i in enumerate(order):
            if s[i] > 0 or s.min() < 0: tot[i] += 1 / (60 + r)
    return tot


def run(q, cfg):
    cs = detect(q)
    lists = [bm25(q), dense(q)]
    if cs and 'terms' in cfg: lists.append(bm25(expanded_terms(cs)))
    if cs and 'classes' in cfg: lists.append(class_score(cs))
    if cs and 'dense_exp' in cfg:
        lists.append(dense(q + '：' + '、'.join(m['name'] for c in cs for m in c['mechanisms'])))
    return rrf(lists), cs


def evaluate(qs, cfg):
    h10 = rec = mrr = 0
    for q, rel in qs:
        s, _ = run(q, cfg)
        top = list(np.argsort(-s)[:10])
        hits = [i for i in top if i in rel]
        h10 += bool(hits); rec += len(hits) / min(len(rel), 10)
        first = next((r for r, i in enumerate(top) if i in rel), None)
        mrr += 0 if first is None else 1 / (first + 1)
    n = len(qs); return h10 / n, rec / n, mrr / n


if __name__ == '__main__':
    jq = [(q, set(J['sets'][k])) for q, k in J['questions']]
    for q, _ in jq:
        assert detect(q), q
    oq = [(q, {t}) for t, q in json.load(open('/root/build/eval_questions.json', encoding='utf-8'))]
    trig = [q for q, _ in oq if detect(q)]
    print('original questions that trigger a concept:', trig)
    cfgs = {'baseline (bm25+dense)': set(), '+ mechanism terms': {'terms'}, '+ class links': {'classes'},
            '+ terms + classes': {'terms', 'classes'}, '+ terms + classes + dense expansion': {'terms', 'classes', 'dense_exp'}}
    print(f"\n{'config':38s} | jargon: hit@10 recall@10 MRR | original: hit@10 MRR")
    for name, cfg in cfgs.items():
        a = evaluate(jq, cfg); b = evaluate(oq, cfg)
        print(f"{name:38s} |   {a[0]:.2f}     {a[1]:.2f}     {a[2]:.2f} |   {b[0]:.2f}   {b[2]:.3f}")
    if len(sys.argv) > 1:
        for q, rel in jq:
            for cfg in [set(), {'terms', 'classes'}]:
                s, cs = run(q, cfg)
                top = list(np.argsort(-s)[:5])
                print(('BASE ' if not cfg else 'GRAPH'), q, '|', ' / '.join(('✓' if i in rel else '·') + docs[i]['title'][:22] for i in top))
