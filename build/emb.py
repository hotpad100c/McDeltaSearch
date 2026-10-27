import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
M='/root/dl/dsh-code-search/package/models/bge-small-zh-v1.5/'
tok=Tokenizer.from_file(M+'tokenizer.json')
so=ort.SessionOptions(); so.intra_op_num_threads=2
sess=ort.InferenceSession(M+'onnx/model_quantized.onnx',so)
names=[i.name for i in sess.get_inputs()]
def embed(texts,maxlen=512):
    tok.enable_truncation(maxlen); tok.enable_padding()
    enc=tok.encode_batch(texts)
    ids=np.array([e.ids for e in enc],dtype=np.int64); am=np.array([e.attention_mask for e in enc],dtype=np.int64)
    feed={'input_ids':ids,'attention_mask':am}
    if 'token_type_ids' in names: feed['token_type_ids']=np.zeros_like(ids)
    out=sess.run(None,feed)[0][:,0]
    return out/np.linalg.norm(out,axis=1,keepdims=True)
