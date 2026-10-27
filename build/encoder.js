
(function (root) {
  'use strict';
  const H = 512, HEADS = 8, HD = 64, FF = 2048, LAYERS = 4;
  const ROLES = ['attention/self/query', 'attention/self/key', 'attention/self/value',
    'attention/output/dense', 'intermediate/dense', 'output/dense'];

  function isCJK(cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
  }
  function isPunct(cp, ch) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(ch);
  }
  function isWS(ch) { return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r' || /\p{Zs}/u.test(ch); }
  function isControl(ch) { if (ch === '\t' || ch === '\n' || ch === '\r') return false; return /\p{C}/u.test(ch); }

  function makeTokenizer(vocabList) {
    const vocab = new Map();
    vocabList.forEach((t, i) => vocab.set(t, i));
    const UNK = vocab.get('[UNK]'), CLS = vocab.get('[CLS]'), SEP = vocab.get('[SEP]');
    function basic(text) {
      let out = '';
      for (const ch of text) {
        const cp = ch.codePointAt(0);
        if (cp === 0 || cp === 0xFFFD || isControl(ch)) continue;
        if (isWS(ch)) { out += ' '; continue; }
        if (isCJK(cp)) { out += ' ' + ch + ' '; continue; }
        out += ch;
      }
      const words = [];
      for (const w of out.split(' ')) {
        if (!w) continue;
        let cur = '';
        for (const ch of w) {
          if (isPunct(ch.codePointAt(0), ch)) { if (cur) words.push(cur); words.push(ch); cur = ''; }
          else cur += ch;
        }
        if (cur) words.push(cur);
      }
      return words;
    }
    function wordpiece(word) {
      const chars = Array.from(word);
      if (chars.length > 100) return [UNK];
      const ids = [];
      let start = 0;
      while (start < chars.length) {
        let end = chars.length, found = -1;
        while (start < end) {
          let sub = chars.slice(start, end).join('');
          if (start > 0) sub = '##' + sub;
          const id = vocab.get(sub);
          if (id !== undefined) { found = id; break; }
          end--;
        }
        if (found < 0) return [UNK];
        ids.push(found);
        start = end;
      }
      return ids;
    }
    return function encode(text, maxLen) {
      const ids = [CLS];
      for (const w of basic(text)) {
        for (const id of wordpiece(w)) { if (ids.length < maxLen - 1) ids.push(id); }
      }
      ids.push(SEP);
      return ids;
    };
  }

  function layerNorm(x, T, w, b) {
    for (let t = 0; t < T; t++) {
      const o = t * H;
      let m = 0; for (let i = 0; i < H; i++) m += x[o + i]; m /= H;
      let v = 0; for (let i = 0; i < H; i++) { const d = x[o + i] - m; v += d * d; } v /= H;
      const r = 1 / Math.sqrt(v + 1e-12);
      for (let i = 0; i < H; i++) x[o + i] = (x[o + i] - m) * r * w[i] + b[i];
    }
  }

  function linear(x, T, W, b, inD, outD) {
    const y = new Float32Array(T * outD);
    for (let t = 0; t < T; t++) {
      const yo = t * outD, xo = t * inD;
      for (let j = 0; j < outD; j++) y[yo + j] = b[j];
      for (let k = 0; k < inD; k++) {
        const xv = x[xo + k]; if (xv === 0) continue;
        const wo = k * outD;
        for (let j = 0; j < outD; j++) y[yo + j] += xv * W[wo + j];
      }
    }
    return y;
  }
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function b64ToBytes(s, out, offset) {
    const bin = atob(s);
    for (let i = 0; i < bin.length; i++) out[offset + i] = bin.charCodeAt(i);
    return bin.length;
  }
  async function fetchBlob(base, manifest, name, onChars) {
    const info = manifest[name];
    const bytes = new Uint8Array(Math.floor(info.chars / 4) * 3);
    let off = 0;
    for (const part of info.parts) {
      const r = await fetch(base + part);
      if (!r.ok) throw new Error('无法加载数据文件 ' + part + '（HTTP ' + r.status + '）');
      const s = await r.text();
      onChars && onChars(s.length);
      off += b64ToBytes(s, bytes, off);
    }

    return bytes.buffer.slice(0, off);
  }
  async function load(base, onProgress) {
    let got = 0;
    const tick = (n) => { got += n; onProgress && onProgress(got); };
    const [man, blobs] = await Promise.all([
      fetch(base + 'model.json').then((r) => { if (!r.ok) throw new Error('model.json HTTP ' + r.status); return r.json(); }),
      fetch(base + 'blobs.json').then((r) => { if (!r.ok) throw new Error('blobs.json HTTP ' + r.status); return r.json(); }),
    ]);
    const [embBuf, layBuf] = await Promise.all([fetchBlob(base, blobs, 'emb', tick), fetchBlob(base, blobs, 'layers', tick)]);
    const bufs = { emb: embBuf, layers: layBuf };
    const T = {};
    for (const t of man.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      const C = t.dtype === 'uint8' ? Uint8Array : t.dtype === 'int8' ? Int8Array : Float32Array;
      T[t.name] = new C(bufs[t.file], t.offset, n);
    }
    const W = { layers: [] };
    W.word = T.word_q; W.wordScale = man.word_scale; W.wordZp = man.word_zp;
    W.pos = T.position; W.posScale = man.pos_scale; W.posZp = man.pos_zp;
    W.tt = T.token_type; W.elnW = T.eln_w; W.elnB = T.eln_b;
    for (let L = 0; L < LAYERS; L++) {
      const layer = {};
      for (const r of ROLES) {
        const q = T[`${L}/${r}/q`], s = T[`${L}/${r}/s`], z = T[`${L}/${r}/z`];
        const outD = s.length, inD = q.length / outD;
        const f = new Float32Array(q.length);
        for (let k = 0; k < inD; k++) for (let j = 0; j < outD; j++) f[k * outD + j] = (q[k * outD + j] - z[j]) * s[j];
        layer[r] = { W: f, b: T[`${L}/${r}/b`], inD, outD };
      }
      layer.ln1 = [T[`${L}/attention.output.LayerNorm/w`], T[`${L}/attention.output.LayerNorm/b`]];
      layer.ln2 = [T[`${L}/output.LayerNorm/w`], T[`${L}/output.LayerNorm/b`]];
      W.layers.push(layer);
    }
    const tokenize = makeTokenizer(man.vocab);

    function encode(text, maxLen) {
      const ids = tokenize(text, maxLen || 128);
      const Tn = ids.length;
      let x = new Float32Array(Tn * H);
      for (let t = 0; t < Tn; t++) {
        const wo = ids[t] * H, po = t * H;
        for (let i = 0; i < H; i++) {
          x[t * H + i] = (W.word[wo + i] - W.wordZp) * W.wordScale + (W.pos[po + i] - W.posZp) * W.posScale + W.tt[i];
        }
      }
      layerNorm(x, Tn, W.elnW, W.elnB);
      const scale = 1 / Math.sqrt(HD);
      for (const layer of W.layers) {
        const lin = (inp, r) => linear(inp, Tn, layer[r].W, layer[r].b, layer[r].inD, layer[r].outD);
        const q = lin(x, ROLES[0]), k = lin(x, ROLES[1]), v = lin(x, ROLES[2]);
        const ctx = new Float32Array(Tn * H);
        const sc = new Float32Array(Tn);
        for (let h = 0; h < HEADS; h++) {
          const ho = h * HD;
          for (let i = 0; i < Tn; i++) {
            let mx = -Infinity;
            for (let j = 0; j < Tn; j++) {
              let d = 0; const qi = i * H + ho, kj = j * H + ho;
              for (let e = 0; e < HD; e++) d += q[qi + e] * k[kj + e];
              d *= scale; sc[j] = d; if (d > mx) mx = d;
            }
            let sum = 0;
            for (let j = 0; j < Tn; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
            const co = i * H + ho;
            for (let j = 0; j < Tn; j++) {
              const p = sc[j] / sum, vj = j * H + ho;
              for (let e = 0; e < HD; e++) ctx[co + e] += p * v[vj + e];
            }
          }
        }
        const ao = lin(ctx, ROLES[3]);
        for (let i = 0; i < x.length; i++) x[i] += ao[i];
        layerNorm(x, Tn, layer.ln1[0], layer.ln1[1]);
        const hdn = lin(x, ROLES[4]);
        for (let i = 0; i < hdn.length; i++) { const a = hdn[i]; hdn[i] = 0.5 * a * (1 + erf(a / Math.SQRT2)); }
        const fo = lin(hdn, ROLES[5]);
        for (let i = 0; i < x.length; i++) x[i] += fo[i];
        layerNorm(x, Tn, layer.ln2[0], layer.ln2[1]);
      }
      const out = new Float32Array(H);
      let n = 0;
      for (let i = 0; i < H; i++) { out[i] = x[i]; n += x[i] * x[i]; }
      n = Math.sqrt(n);
      for (let i = 0; i < H; i++) out[i] /= n;
      return { vec: out, tokens: Tn };
    }
    return { encode, tokenize };
  }
  root.MCEncoder = { load, fetchBlob };
})(typeof window !== 'undefined' ? window : globalThis);
