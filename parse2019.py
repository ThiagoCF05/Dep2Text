__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Script for parsing the original corpus before apply the alignment method

PYTHON VERSION: 2.7
"""

import os
import json


def parse_dep(snt):
    root, nodes, edges = 1, {}, {}

    for elem in snt:
        elem = elem.split('\t')

        id = elem[0]
        word = elem[1]
        lemma = elem[2]
        upos = elem[3]
        xpos = elem[4]
        feats = {}
        if elem[5] != '_':
            feats = dict(map(lambda x: (x.split('=')[0], x.split('=')[1]), elem[5].split('|')))
        head = elem[6]
        deps = elem[7]
        _ = elem[8]
        _ = elem[9]

        # node = Node(id=id, lemma=lemma, upos=upos, xpos=xpos, feats=feats, head=head, deps=deps)
        node = {
            'id':id,
            'lemma':lemma,
            'realization': word,
            'order_id': id,
            'upos':upos,
            'xpos':xpos,
            'feats':feats,
            'head':head,
            'deps':deps
        }
        nodes[id] = node
        edges[id] = []

        if deps == 'root':
            root = id

    for node in nodes:
        head = nodes[node]['head']
        deps = nodes[node]['deps']
        if head not in ['0', '_']:
            edges[head].append({'deps':deps, 'node':node})

    # tree = Tree(nodes=nodes, edges=edges, root=root)
    tree = {'root':root, 'nodes':nodes, 'edges':edges}
    return tree


def parse(path, lngs):
    languages = {}

    for fname in [w for w in os.listdir(path) if not str(w).startswith('.')]:
        lng = fname.split('_')[0]
        if lng in lngs:
            if lng not in languages:
                languages[lng] = []

            print(fname, lng)
            with open(os.path.join(path, fname)) as f:
                doc = f.read()
            doc = doc.split('\n\n')

            for inst in doc[:-1]:
                attrs = {}
                rows = inst.split('\n')
                try:
                    for i, elem in enumerate(rows):
                        if elem[0] == '#':
                            try:
                                _id, value = elem.split('=', 1)
                                attrs[_id.replace('#', '').strip()] = value.strip()
                            except:
                                pass
                        else:
                            attrs['tree'] = parse_dep(rows[i:])
                            languages[lng].append(attrs)
                            break
                except:
                    print('error')
    return languages

def to_json(fname, languageset):
    json.dump(languageset, open(fname, 'w'))

if __name__ == '__main__':
    PATH = 'data2019/UD_train-dev'
    TRAIN_PATH = os.path.join(PATH, 'UD-train')
    DEV_PATH = os.path.join(PATH, 'UD-dev')
    # TEST_DEP_PATH = os.path.join(DEP_PATH, 'test')

    JSON_PATH = 'data2019/json'
    if not os.path.exists(JSON_PATH):
        os.mkdir(JSON_PATH)

    trainset = parse(TRAIN_PATH, ['ar', 'en', 'es', 'fr', 'hi', 'id', 'ja', 'ko', 'pt', 'ru', 'zh'])
    TRAIN_SAVE_PATH = 'data2019/json/train'
    if not os.path.exists(TRAIN_SAVE_PATH):
        os.mkdir(TRAIN_SAVE_PATH)
    for lng, languageset in trainset.items():
        fname = os.path.join(TRAIN_SAVE_PATH, lng + '.json')
        to_json(fname, languageset)

    devset = parse(DEV_PATH, ['ar', 'en', 'es', 'fr', 'hi', 'id', 'ja', 'ko', 'pt', 'ru', 'zh'])
    DEV_SAVE_PATH = 'data2019/json/dev'
    if not os.path.exists(DEV_SAVE_PATH):
        os.mkdir(DEV_SAVE_PATH)
    for lng, languageset in devset.items():
        fname = os.path.join(DEV_SAVE_PATH, lng + '.json')
        to_json(fname, languageset)

    # testset = parse(TEST_DEP_PATH, TEST_SNT_PATH)
    # TEST_SAVE_PATH = 'data2019/json/test'
    # if not os.path.exists(TEST_SAVE_PATH):
    #     os.mkdir(TEST_SAVE_PATH)
    # for lng, languageset in testset.iteritems():
    #     fname = os.path.join(TEST_SAVE_PATH, lng + '.json')
    #     to_json(fname, languageset)