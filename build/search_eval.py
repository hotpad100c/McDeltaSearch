import sys, json, re, math, collections, numpy as np
sys.path.insert(0, '/root/build')
from model import float_weights, forward
from emb import tok

c = json.load(open('/root/build/corpus.json', encoding='utf-8'))
docs, versions = c['docs'], c['versions']
vecs = np.load('/root/build/vecs.npy'); owner = np.load('/root/build/owner.npy')
W, _, _ = float_weights()
INSTR = '为这个句子生成表示以用于检索相关文章：'

CJK = re.compile(r'[㐀-鿿豈-﫿]+')
WORD = re.compile(r'[A-Za-z0-9_.$]+')


def terms(text):
    out = []
    for run in CJK.findall(text):
        if len(run) == 1:
            out.append(run)
        out += [run[i:i + 2] for i in range(len(run) - 1)]
    out += [w.lower().strip('.') for w in WORD.findall(text) if len(w.strip('.')) > 1]
    return out


N = len(docs); df = collections.Counter(); tfs = []; lens = []
for d in docs:
    t = terms(d['title']) * 3 + terms(' '.join(d['classes'])) * 2 + terms(' '.join(h + ' ' + x for h, x in d['parts']))
    tf = collections.Counter(t); tfs.append(tf); lens.append(len(t)); df.update(tf.keys())
avg = sum(lens) / N


def bm25(q, k1=1.2, b=0.75):
    qt = set(terms(q)); s = np.zeros(N)
    for t in qt:
        if t not in df: continue
        idf = math.log(1 + (N - df[t] + .5) / (df[t] + .5))
        for i, tf in enumerate(tfs):
            f = tf.get(t)
            if f: s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
    return s


def dense(q):
    ids = tok.encode(INSTR + q).ids
    qv = forward(W, np.array([ids]), np.ones((1, len(ids)), np.float32))[0]
    s = vecs @ qv; best = np.full(N, -1.0)
    np.maximum.at(best, owner, s)
    return best


def search(q, k=8):
    b, d = bm25(q), dense(q)
    rb = {i: r for r, i in enumerate(np.argsort(-b)[:200]) if b[i] > 0}
    rd = {i: r for r, i in enumerate(np.argsort(-d)[:200])}
    sc = collections.Counter()
    for i, r in rb.items(): sc[i] += 1 / (60 + r)
    for i, r in rd.items(): sc[i] += 1 / (60 + r)
    res = []
    for i, s in sc.most_common(k):
        res.append((versions[docs[i]['v']]['version'], docs[i]['title'], rd.get(i), rb.get(i)))
    return res


if __name__ == '__main__':
    for q in sys.argv[1:]:
        print('##', q)
        for r in search(q):
            print('  ', r)
