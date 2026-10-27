import sys, os, json, base64, shutil

src, concepts_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
CH = 8 * 1024 * 1024
man = {}
for name in ['emb', 'layers', 'vectors']:
    s = base64.b64encode(open(f'{src}/{name}.bin', 'rb').read()).decode()
    parts = []
    for i in range(0, len(s), CH):
        fn = f'{name}.{i // CH}.txt'
        open(f'{out}/{fn}', 'w').write(s[i:i + CH]); parts.append(fn)
    man[name] = {'parts': parts, 'chars': len(s)}
json.dump(man, open(f'{out}/blobs.json', 'w'))
for f in ['docs.json', 'model.json']:
    shutil.copy(f'{src}/{f}', f'{out}/{f}')

g = json.load(open(concepts_path, encoding='utf-8'))
web = {'mechanisms': g['mechanisms'],
       'concepts': [{k: c[k] for k in ['id', 'name', 'category', 'desc', 'aliases', 'role', 'mechanisms', 'confidence']} for c in g['concepts']]}
json.dump(web, open(f'{out}/concepts.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))

here = os.path.dirname(os.path.abspath(__file__))
page = open(f'{here}/page.html', encoding='utf-8').read()
enc = open(f'{here}/encoder.js', encoding='utf-8').read()
html = ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        '<style>[hidden]{display:none!important}body{margin:0}</style>\n</head>\n<body>\n'
        + page.replace('/*ENCODER*/', enc) + '\n</body>\n</html>\n')
open(f'{out}/index.html', 'w', encoding='utf-8').write(html)
print('site written to', out)
