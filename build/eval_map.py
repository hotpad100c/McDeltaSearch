import sys, json, numpy as np
sys.path.insert(0, '/root/build')
from search_eval import bm25, docs, N, INSTR, W as SW
from model import forward
from emb import tok as stok, embed as sembed_ort
import embed_large_q as L

qs = json.load(open('/root/build/eval_questions.json', encoding='utf-8'))
owner = np.load('/root/build/owner.npy')
vs = np.load('/root/build/vecs.npy')            # small doc windows
vl = np.load('/root/build/vecs_large.npy')      # large doc windows
tl = np.load('/root/build/titles_large.npy')    # large, INSTR+title
ts = np.load('/root/build/titles_small.npy')    # small (ORT), INSTR+title


def small_q(q):  # exactly what the browser computes
    stok.no_padding(); stok.no_truncation()
    ids = stok.encode(INSTR + q).ids
    return forward(SW, np.array([ids]), np.ones((1, len(ids)), np.float32))[0]


def docmax(vecs, qv):
    s = vecs @ qv; best = np.full(N, -1.0); np.maximum.at(best, owner, s); return best


def rrf(*scores):
    tot = np.zeros(N)
    for s in scores:
        order = np.argsort(-s)[:150]
        for r, i in enumerate(order):
            if s[i] > 0 or s is not scores[0]: tot[i] += 1 / (60 + r)
    return tot


def metrics(ranks):
    r = np.array(ranks)
    return dict(h1=np.mean(r < 1), h5=np.mean(r < 5), h10=np.mean(r < 10), mrr=np.mean([1 / (x + 1) if x < 10 else 0 for x in r]))


def rank_of(score, target):
    return int((score > score[target]).sum())


# ---- fit linear map small -> large on corpus texts (windows + titles) ----
X = np.vstack([vs, ts]); Y = np.vstack([vl, tl])
Xb = np.hstack([X, np.ones((len(X), 1))])
maps = {}
for lam in [0.01, 0.1, 1.0]:
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); maps[lam] = np.linalg.solve(A, Xb.T @ Y)


def mapped(qv, lam):
    y = np.append(qv, 1) @ maps[lam]; return y / np.linalg.norm(y)


qsm = [small_q(q) for _, q in qs]
qlg = L.embed([INSTR + q for _, q in qs])
bms = [bm25(q) for _, q in qs]

configs = {
    'small': lambda k: docmax(vs, qsm[k]),
    **{f'small→map(λ={lam})→large': (lambda lam: lambda k: docmax(vl, mapped(qsm[k], lam)))(lam) for lam in maps},
    'large (upper bound)': lambda k: docmax(vl, qlg[k]),
}
print(f'{len(qs)} questions\n')
print(f"{'config':28s} {'mode':8s} hit@1  hit@5  hit@10  MRR@10")
for name, fn in configs.items():
    for mode in ['dense', 'hybrid']:
        ranks = []
        for k, (t, q) in enumerate(qs):
            d = fn(k)
            s = d if mode == 'dense' else rrf(bms[k], d)
            ranks.append(rank_of(s, t))
        m = metrics(ranks)
        print(f"{name:28s} {mode:8s} {m['h1']:.2f}   {m['h5']:.2f}   {m['h10']:.2f}    {m['mrr']:.3f}")
b = metrics([rank_of(bms[k], t) for k, (t, q) in enumerate(qs)])
print(f"{'keyword only':28s} {'bm25':8s} {b['h1']:.2f}   {b['h5']:.2f}   {b['h10']:.2f}    {b['mrr']:.3f}")
