import sys, json, time, numpy as np
sys.path.insert(0, '/root/build')
from emb import embed

c = json.load(open('/root/build/corpus.json', encoding='utf-8'))
WIN = 440
texts, owner = [], []
for i, d in enumerate(c['docs']):
    body = '\n'.join((h + '：' if h else '') + t for h, t in d['parts'])
    chunks = [body[k:k + WIN] for k in range(0, max(len(body), 1), WIN - 40)][:3]
    if len(chunks) > 1 and len(chunks[-1]) < 120:
        chunks = chunks[:-1]
    for ch in chunks:
        texts.append(d['title'] + '\n' + ch)
        owner.append(i)
print(len(texts), 'windows', flush=True)
order = np.argsort([len(t) for t in texts])
out = np.zeros((len(texts), 512), np.float32)
t0 = time.time()
B = 16
for b in range(0, len(order), B):
    idx = order[b:b + B]
    out[idx] = embed([texts[j] for j in idx])
    if b % (B * 40) == 0:
        print(b, round(time.time() - t0), flush=True)
np.save('/root/build/vecs.npy', out)
np.save('/root/build/owner.npy', np.array(owner, np.int32))
print('done', round(time.time() - t0))
