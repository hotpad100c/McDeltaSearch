import numpy as np, onnxruntime as ort
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
