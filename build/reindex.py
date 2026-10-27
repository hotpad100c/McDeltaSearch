import sys, json, hashlib, subprocess, time, numpy as np
sys.path.insert(0, '.')

prev_corpus, prev_vecs, prev_owner = sys.argv[1:4]
WIN = 440


def key(c, d):
    return c['versions'][d['v']]['version'] + '#' + d['slug']


def windows(d):
    body = '\n'.join((h + '：' if h else '') + t for h, t in d['parts'])
    chunks = [body[k:k + WIN] for k in range(0, max(len(body), 1), WIN - 40)][:3]
    if len(chunks) > 1 and len(chunks[-1]) < 120:
        chunks = chunks[:-1]
    return [d['title'] + '\n' + ch for ch in chunks]


def digest(d):
    return hashlib.sha1(json.dumps([d['title'], d['parts']], ensure_ascii=False).encode()).hexdigest()


subprocess.run([sys.executable, 'parse.py'], check=True)
old = json.load(open(prev_corpus, encoding='utf-8'))
new = json.load(open('corpus.json', encoding='utf-8'))
ov, oo = np.load(prev_vecs), np.load(prev_owner)
old_rows = {}
for i, d in enumerate(old['docs']):
    old_rows[(key(old, d), digest(d))] = np.where(oo == i)[0]

vecs, owner, todo = [], [], []
for i, d in enumerate(new['docs']):
    rows = old_rows.get((key(new, d), digest(d)))
    if rows is not None and len(rows):
        for r in rows:
            vecs.append(ov[r]); owner.append(i)
    else:
        for t in windows(d):
            todo.append((len(vecs), t)); vecs.append(None); owner.append(i)

print(f'sections: {len(new["docs"])} (old {len(old["docs"])}); windows to embed: {len(todo)}', flush=True)
if todo:
    from emb import embed
    t0 = time.time()
    order = sorted(todo, key=lambda x: len(x[1]))
    for b in range(0, len(order), 16):
        batch = order[b:b + 16]
        out = embed([t for _, t in batch])
        for (slot, _), v in zip(batch, out):
            vecs[slot] = v
    print(f'embedded {len(todo)} windows in {time.time() - t0:.0f}s')
np.save('vecs.npy', np.vstack(vecs).astype(np.float32))
np.save('owner.npy', np.array(owner, np.int32))
gone = set(k for k, _ in old_rows) - set(key(new, d) for d in new['docs'])
print('removed sections:', len(gone))
subprocess.run([sys.executable, 'export_data.py'], check=True)
