import numpy as np, onnx, json, os
from onnx import numpy_helper
from math import erf
SRC='/root/dl/dsh-code-search/package/models/bge-small-zh-v1.5/'
def load_parts():
    m=onnx.load(SRC+'onnx/model_quantized.onnx'); g=m.graph
    I={t.name:numpy_helper.to_array(t) for t in g.initializer}
    mm={}
    for n in g.node:
        if n.op_type=='MatMulInteger':
            p=n.name.split('/'); L=int(p[2].split('.')[1]); role='/'.join(p[3:-1])
            mm[(L,role)]=n.input[1].replace('_quantized','')
    return I,mm
ROLES=['attention/self/query','attention/self/key','attention/self/value','attention/output/dense','intermediate/dense','output/dense']
BN={'attention/self/query':'attention.self.query','attention/self/key':'attention.self.key','attention/self/value':'attention.self.value','attention/output/dense':'attention.output.dense','intermediate/dense':'intermediate.dense','output/dense':'output.dense'}
def float_weights():
    I,mm=load_parts(); W={}
    for e in ['word','position','token_type']:
        k=f'embeddings.{e}_embeddings.weight'
        W[e]=(I[k+'_quantized'].astype(np.float32)-I[k+'_zero_point'].astype(np.float32))*I[k+'_scale']
    W['eln_w']=I['embeddings.LayerNorm.weight'];W['eln_b']=I['embeddings.LayerNorm.bias']
    for L in range(4):
        for r in ROLES:
            b=mm[(L,r)]
            W[(L,r)]=(I[b+'_quantized'].astype(np.float32)-I[b+'_zero_point'].astype(np.float32))*I[b+'_scale']
            W[(L,r,'b')]=I[f'encoder.layer.{L}.{BN[r]}.bias']
        for ln in ['attention.output.LayerNorm','output.LayerNorm']:
            W[(L,ln,'w')]=I[f'encoder.layer.{L}.{ln}.weight'];W[(L,ln,'b')]=I[f'encoder.layer.{L}.{ln}.bias']
    return W,I,mm
def ln(x,w,b,eps=1e-12):
    m=x.mean(-1,keepdims=True);v=((x-m)**2).mean(-1,keepdims=True);return (x-m)/np.sqrt(v+eps)*w+b
from scipy.special import erf as verf
def forward(W,ids,mask,H=8):
    B,T=ids.shape
    x=W['word'][ids]+W['position'][np.arange(T)][None]+W['token_type'][0][None,None]
    x=ln(x,W['eln_w'],W['eln_b'])
    bias=(1-mask[:,None,None,:])*-1e9
    for L in range(4):
        q=x@W[(L,ROLES[0])]+W[(L,ROLES[0],'b')];k=x@W[(L,ROLES[1])]+W[(L,ROLES[1],'b')];v=x@W[(L,ROLES[2])]+W[(L,ROLES[2],'b')]
        sp=lambda t:t.reshape(B,T,H,-1).transpose(0,2,1,3)
        q,k,v=sp(q),sp(k),sp(v)
        a=q@k.transpose(0,1,3,2)/np.sqrt(q.shape[-1])+bias
        a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
        o=(a@v).transpose(0,2,1,3).reshape(B,T,-1)
        o=o@W[(L,ROLES[3])]+W[(L,ROLES[3],'b')]
        x=ln(x+o,W[(L,'attention.output.LayerNorm','w')],W[(L,'attention.output.LayerNorm','b')])
        h=x@W[(L,ROLES[4])]+W[(L,ROLES[4],'b')];h=0.5*h*(1+verf(h/np.sqrt(2)))
        h=h@W[(L,ROLES[5])]+W[(L,ROLES[5],'b')]
        x=ln(x+h,W[(L,'output.LayerNorm','w')],W[(L,'output.LayerNorm','b')])
    c=x[:,0];return c/np.linalg.norm(c,axis=1,keepdims=True)
