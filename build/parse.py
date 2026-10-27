import os, re, json, glob, collections

REPO = '/root/mcdelta'
COMMIT = os.popen(f'git -C {REPO} rev-parse HEAD').read().strip()

LINK = re.compile(r'\[([^\]]*)\]\(([^)]*)\)')
JAVA = re.compile(r'/([A-Za-z0-9_$]+)\.java')
SRC = re.compile(r'\]\((?:https://github\.com/TISUnion/guardian/blob/[0-9a-f]+/)?(?:src/main/(?:java|resources)/)?([^)#]+)')


def vkey(name):
    return tuple(int(x) for x in re.findall(r'\d+', name))


def slugify(text, seen):
    s = text.strip().lower()
    s = re.sub(r'[^\w\- ]', '', s)
    s = s.replace(' ', '-')
    n = seen[s]
    seen[s] += 1
    return s if n == 0 else f'{s}-{n}'


def clean(text):
    text = LINK.sub(r'\1', text)
    text = text.replace('`', '')
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return text


FULL_HEADS = ('差异出现的场景', '版本变化', '实际影响')


def section_form(parts):
    if any(h in FULL_HEADS for h, _ in parts):
        return 'full'
    lines = [l for l in '\n'.join(t for _, t in parts).split('\n') if l.strip()]
    bullets = [l for l in lines if l.lstrip().startswith('- ')]
    if len(bullets) >= 2 and lines and lines[0].lstrip().startswith('- '):
        return 'merged'
    return 'single'


def parse_file(path):
    rel = os.path.basename(os.path.dirname(path))
    lines = open(path, encoding='utf-8').read().split('\n')
    prev = cur = None
    m = re.match(r'#\s+Minecraft\s+(.*?)\s+变更分析', lines[0])
    title_ver = m.group(1) if m else os.path.basename(path)[:-3]
    sections, cursec = [], None
    seen = collections.Counter()
    for ln in lines:
        if ln.startswith('- 上一个版本：') and prev is None:
            prev = ln.split('：', 1)[1].strip().strip('`')
        elif ln.startswith('- 当前版本：') and cur is None:
            cur = re.match(r'`?([^`（(]+)`?', ln.split('：', 1)[1].strip()).group(1).strip()
        if ln.startswith('## '):
            cursec = {'title': ln[3:].strip(), 'lines': []}
            cursec['slug'] = slugify(cursec['title'], seen)
            sections.append(cursec)
        elif ln.startswith('### '):
            slugify(ln[4:], seen)
            if cursec:
                cursec['lines'].append(ln)
        elif cursec is not None:
            cursec['lines'].append(ln)
    out = []
    for i, s in enumerate(sections):
        classes, paths, body = [], [], []
        for ln in s['lines']:
            if ln.startswith('代码出处'):
                for c in JAVA.findall(ln):
                    if c not in classes:
                        classes.append(c)
                for x in SRC.findall(ln):
                    if x not in paths:
                        paths.append(x)
                continue
            body.append(ln)
        parts, curh, buf = [], '', []
        for ln in body:
            if ln.startswith('### '):
                if buf or curh:
                    parts.append([curh, '\n'.join(buf).strip()])
                curh, buf = ln[4:].strip(), []
            else:
                buf.append(ln)
        parts.append([curh, '\n'.join(buf).strip()])
        parts = [[h, clean(t)] for h, t in parts if t or h]
        out.append({
            'title': clean(s['title']), 'slug': s['slug'], 'parts': parts, 'classes': classes, 'paths': paths,
            'form': section_form(parts), 'rank': i + 1,
        })
    return {'release': rel, 'file': os.path.basename(path)[:-3], 'version': cur or title_ver,
            'label': title_ver, 'prev': prev, 'sections': out}


def order(files):
    by_rel = collections.defaultdict(list)
    for f in files:
        by_rel[f['release']].append(f)
    ordered = []
    for rel in sorted(by_rel, key=vkey):
        fs = by_rel[rel]
        names = {f['version']: f for f in fs}
        children = collections.defaultdict(list)
        roots = []
        for f in fs:
            if f['prev'] in names:
                children[f['prev']].append(f)
            else:
                roots.append(f)
        done = set()

        def walk(f):
            if f['version'] in done:
                return
            done.add(f['version'])
            ordered.append(f)
            for c in sorted(children[f['version']], key=lambda x: x['file']):
                walk(c)
        for r in sorted(roots, key=lambda x: x['file']):
            walk(r)
        for f in sorted(fs, key=lambda x: x['file']):
            walk(f)
    return ordered


if __name__ == '__main__':
    files = [parse_file(p) for p in glob.glob(f'{REPO}/reports/*/*.md')]
    files = order(files)
    docs = []
    for vi, f in enumerate(files):
        for si, s in enumerate(f['sections']):
            docs.append({'v': vi, 'title': s['title'], 'slug': s['slug'], 'parts': s['parts'],
                         'classes': s['classes'], 'paths': s['paths'], 'form': s['form'], 'rank': s['rank']})
    versions = [{'release': f['release'], 'file': f['file'], 'version': f['version'],
                 'label': f['label'], 'prev': f['prev'], 'n': len(f['sections'])} for f in files]
    json.dump({'commit': COMMIT, 'versions': versions, 'docs': docs},
              open('/root/build/corpus.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(len(versions), 'versions', len(docs), 'sections')
    import collections as _c
    print('forms:', dict(_c.Counter(d['form'] for d in docs)))
    print([v['version'] for v in versions[:12]], [v['version'] for v in versions[-6:]])
    lens = sorted(sum(len(t) for _, t in d['parts']) for d in docs)
    print('body chars median', lens[len(lens) // 2], 'p90', lens[int(len(lens) * .9)], 'max', lens[-1],
          'total', sum(lens))
