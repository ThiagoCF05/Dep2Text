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

def parse_text(path):
    with open(path) as f:
        doc = f.read().decode('utf-8')

    texts = []
    doc = doc.split('\n\n')

    for row in doc:
        text, sent_id = '', ''

        for elem in row.split('\n'):
            if '# text =' in elem:
                text = elem.replace('# text =', '').strip()
            if '# sent_id =' in elem:
                sent_id = elem.replace('# sent_id =', '').strip()

        # text_ = Text(sent_id=sent_id, text=text)
        text_ = {'sent_id':sent_id, 'text':text}
        texts.append(text_)

    return texts

def parse_dep(path):
    trees = []

    with open(path) as f:
        doc = f.read().decode('utf-8')

    sentences = doc.split('\n\n')[:-1]

    for i, snt in enumerate(sentences):
        root, nodes, edges = 1, {}, {}
        snt = snt.split('\n')

        for elem in snt:
            elem = elem.split('\t')

            id = elem[0]
            lemma = elem[1]
            _ = elem[2]
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
            if head != '0':
                edges[head].append({'deps':deps, 'node':node})

        # tree = Tree(nodes=nodes, edges=edges, root=root)
        tree = {'root':root, 'nodes':nodes, 'edges':edges}
        trees.append(tree)
    return trees

def parse(dep_path, snt_path):
    instances = {}

    for fname in os.listdir(dep_path):
        if fname != '.DS_Store':
            lng = fname.split('-')[0].split('_')[0]
            if lng in ['en', 'es', 'fr', 'it', 'nl', 'pt']:
                print fname, lng
                trees_ = parse_dep(os.path.join(dep_path, fname))
                snt_fname = fname.replace('.conll', '_sentences.txt')
                texts_ = parse_text(os.path.join(snt_path, snt_fname))

                lng_instances = []
                for i, tree in enumerate(trees_):
                    sent_id, text = texts_[i]['sent_id'], texts_[i]['text']
                    lng_instances.append({'tree':tree, 'sent_id': sent_id, 'text':text})

                instances[lng] = lng_instances

    return instances

def to_json(fname, languageset):
    json.dump(languageset, open(fname, 'w'))

if __name__ == '__main__':
    DEP_PATH = 'data/T1-input'
    TRAIN_DEP_PATH = os.path.join(DEP_PATH, 'train')
    DEV_DEP_PATH = os.path.join(DEP_PATH, 'dev')
    TEST_DEP_PATH = os.path.join(DEP_PATH, 'test')

    SNT_PATH = 'data/Sentences'
    TRAIN_SNT_PATH = os.path.join(SNT_PATH, 'train')
    DEV_SNT_PATH = os.path.join(SNT_PATH, 'dev')
    TEST_SNT_PATH = os.path.join(SNT_PATH, 'test')

    JSON_PATH = 'data/json'
    if not os.path.exists(JSON_PATH):
        os.mkdir(JSON_PATH)

    trainset = parse(TRAIN_DEP_PATH, TRAIN_SNT_PATH)
    TRAIN_SAVE_PATH = 'data/json/train'
    if not os.path.exists(TRAIN_SAVE_PATH):
        os.mkdir(TRAIN_SAVE_PATH)
    for lng, languageset in trainset.iteritems():
        fname = os.path.join(TRAIN_SAVE_PATH, lng + '.json')
        to_json(fname, languageset)

    devset = parse(DEV_DEP_PATH, DEV_SNT_PATH)
    DEV_SAVE_PATH = 'data/json/dev'
    if not os.path.exists(DEV_SAVE_PATH):
        os.mkdir(DEV_SAVE_PATH)
    for lng, languageset in devset.iteritems():
        fname = os.path.join(DEV_SAVE_PATH, lng + '.json')
        to_json(fname, languageset)

    testset = parse(TEST_DEP_PATH, TEST_SNT_PATH)
    TEST_SAVE_PATH = 'data/json/test'
    if not os.path.exists(TEST_SAVE_PATH):
        os.mkdir(TEST_SAVE_PATH)
    for lng, languageset in testset.iteritems():
        fname = os.path.join(TEST_SAVE_PATH, lng + '.json')
        to_json(fname, languageset)