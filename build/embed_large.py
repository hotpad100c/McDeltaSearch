import sys, json, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/root/large/tokenizer.json')
so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession('/root/large/model_quantized.onnx', so)
def embed(texts, maxlen=512):
    tok.enable_truncation(maxlen); tok.enable_padding()
    enc = tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], np.int64); am = np.array([e.attention_mask for e in enc], np.int64)
    out = sess.run(None, {'input_ids': ids, 'attention_mask': am, 'token_type_ids': np.zeros_like(ids)})[0][:, 0]
    return out / np.linalg.norm(out, axis=1, keepdims=True)
INSTR = '为这个句子生成表示以用于检索相关文章：'
c = json.load(open('/root/build/corpus.json', encoding='utf-8'))
WIN = 440
texts = []
for d in c['docs']:
    body = '\n'.join((h + '：' if h else '') + t for h, t in d['parts'])
    chunks = [body[k:k + WIN] for k in range(0, max(len(body), 1), WIN - 40)][:3]
    if len(chunks) > 1 and len(chunks[-1]) < 120: chunks = chunks[:-1]
    texts += [d['title'] + '\n' + ch for ch in chunks]
titles = [INSTR + d['title'] for d in c['docs']]
import os
def run(tx, path):
    order = np.argsort([len(t) for t in tx]); t0 = time.time()
    ck = path + '.partial.npz'
    if os.path.exists(ck):
        z = np.load(ck); out = z['out']; start = int(z['next'])
    else:
        out = np.zeros((len(tx), 1024), np.float32); start = 0
    for b in range(start, len(order), 16):
        idx = order[b:b + 16]; out[idx] = embed([tx[j] for j in idx])
        if b % 320 == 0:
            np.savez(ck, out=out, next=b + 16)
            print(path, b, len(tx), round(time.time() - t0), flush=True)
    np.save(path, out)
if not os.path.exists('/root/build/titles_large.npy'):
    run(titles, '/root/build/titles_large.npy')
run(texts, '/root/build/vecs_large.npy')
print('done', flush=True)
