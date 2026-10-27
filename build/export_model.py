"""Export bge-small-zh-v1.5 int8 ONNX weights + vocab into the layout encoder.js reads.

Input : ONNX + tokenizer.json of bge-small-zh-v1.5 (quantized, transformers.js layout)
Output: OUT/model.json, OUT/emb.bin, OUT/layers.bin
"""
import sys, json, numpy as np
sys.path.insert(0, '.')
from model import load_parts, ROLES, BN, SRC  # SRC points at the model folder; edit model.py to change it

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
I, mm = load_parts()
man = {'tensors': []}
cur = None


def put(buf, name, arr, dtype):
    a = np.ascontiguousarray(arr).astype(dtype)
    off = sum(len(b) for b in buf)
    pad = (-off) % 4
    if pad:
        buf.append(b'\0' * pad); off += pad
    buf.append(a.tobytes())
    man['tensors'].append({'name': name, 'file': cur, 'dtype': str(np.dtype(dtype)), 'shape': list(a.shape), 'offset': off})


cur = 'emb'; buf = []
k = 'embeddings.word_embeddings.weight'
put(buf, 'word_q', I[k + '_quantized'], np.uint8)
man['word_scale'] = float(I[k + '_scale']); man['word_zp'] = int(I[k + '_zero_point'])
open(f'{OUT}/emb.bin', 'wb').write(b''.join(buf))

cur = 'layers'; buf = []
k = 'embeddings.position_embeddings.weight'
put(buf, 'position', I[k + '_quantized'], np.uint8)
man['pos_scale'] = float(I[k + '_scale']); man['pos_zp'] = int(I[k + '_zero_point'])
k = 'embeddings.token_type_embeddings.weight'
put(buf, 'token_type', (I[k + '_quantized'].astype(np.float32) - float(I[k + '_zero_point'])) * float(I[k + '_scale']), np.float32)
put(buf, 'eln_w', I['embeddings.LayerNorm.weight'], np.float32); put(buf, 'eln_b', I['embeddings.LayerNorm.bias'], np.float32)
for L in range(4):
    for r in ROLES:
        b = mm[(L, r)]; n = f'{L}/{r}'
        put(buf, n + '/q', I[b + '_quantized'], np.int8); put(buf, n + '/s', I[b + '_scale'], np.float32)
        put(buf, n + '/z', I[b + '_zero_point'], np.int8); put(buf, n + '/b', I[f'encoder.layer.{L}.{BN[r]}.bias'], np.float32)
    for ln in ['attention.output.LayerNorm', 'output.LayerNorm']:
        put(buf, f'{L}/{ln}/w', I[f'encoder.layer.{L}.{ln}.weight'], np.float32)
        put(buf, f'{L}/{ln}/b', I[f'encoder.layer.{L}.{ln}.bias'], np.float32)
open(f'{OUT}/layers.bin', 'wb').write(b''.join(buf))

vocab = json.load(open(SRC + 'tokenizer.json', encoding='utf-8'))['model']['vocab']
inv = [None] * len(vocab)
for t, i in vocab.items(): inv[i] = t
man['vocab'] = inv
json.dump(man, open(f'{OUT}/model.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print('wrote', OUT)
