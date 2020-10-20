__author__='thiagocastroferreira'

import cPickle as p
import json
import operator
import os

from collections import Counter

class Lexicon:
    def __init__(self, path, out_path):
        self.path = path

        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

    def process(self):
        for language in os.listdir(os.path.join(self.path)):
            print('Language: ', language)
            path = os.path.join(self.path, language)
            procset = json.load(open(path))

            lexicon = {}
            for i, row in enumerate(procset):
                # print('Progress: ', round(i/len(procset), 2), i)
                tree = row['tree']
                nodes = tree['nodes']
                for node in nodes:
                    lemma = nodes[node]['lemma'].lower()
                    features = tuple(nodes[node]['feats'].items())
                    realization = nodes[node]['realization']
                    if len(features) > 0:
                        if lemma not in lexicon:
                            lexicon[lemma] = {}
                        if features not in lexicon[lemma]:
                            lexicon[lemma][features] = []
                        lexicon[lemma][features].append(realization.lower())

            for lemma in lexicon:
                for features in lexicon[lemma]:
                    realization = max(Counter(lexicon[lemma][features]).items(), key=operator.itemgetter(1))[0]
                    lexicon[lemma][features] = realization

            out_path = os.path.join(self.out_path, language)
            p.dump(lexicon, open(out_path, 'wb'))

if __name__ == '__main__':
    in_path = 'data2020/json/train'
    out_path = 'data2020/lexicon'

    lex = Lexicon(path=in_path, out_path=out_path)
    lex.process()