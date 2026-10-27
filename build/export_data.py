import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_category import doc_cat

c = json.load(open('/root/build/corpus.json', encoding='utf-8'))
vecs = np.load('/root/build/vecs.npy'); owner = np.load('/root/build/owner.npy')
n = len(vecs)
scale = np.abs(vecs).max(1) / 127.0
q = np.round(vecs / scale[:, None]).astype(np.int8)
deq = q.astype(np.float32) * scale[:, None]
print('int8 cosine fidelity min', float(((deq / np.linalg.norm(deq, axis=1, keepdims=True)) * vecs).sum(1).min()))
with open('/root/site/vectors.bin', 'wb') as f:
    f.write(scale.astype(np.float32).tobytes())
    o = owner.astype(np.uint16)
    f.write(o.tobytes())
    if (2 * n) % 4: f.write(b'\0\0')
    f.write(q.tobytes())
data = {
    'commit': c['commit'], 'nwin': int(n),
    # versions: [release, file, version, label, number of sections in the report]
    'versions': [[v['release'], v['file'], v['version'], v['label'], v['n']] for v in c['versions']],
    # docs: [version index, title, slug, parts, classes, form (full|single|merged), rank in report (1-based),
    #        category inferred from source paths (path_category.py; null when the paths cannot tell)]
    'docs': [[d['v'], d['title'], d['slug'], d['parts'], d['classes'], d['form'], d['rank'], doc_cat(d['paths'], 'majority')]
             for d in c['docs']],
}
json.dump(data, open('/root/site/docs.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
for f in sorted(os.listdir('/root/site')):
    print(f, os.path.getsize('/root/site/' + f))
