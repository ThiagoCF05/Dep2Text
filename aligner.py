__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Script to align the source lemmas with the target words

PYTHON VERSION: 2.7
DEPENDENCIES:
    spaCy: https://spacy.io
"""

import contractions_pt as contractions
import copy
import os
import json

import spacy

from nltk.metrics import edit_distance

class Aligner(object):
    def __init__(self, ALIGN_PATH, LEXICON_PATH, language):
        self.nlp = spacy.load(language, disable=['ner'])

        instances = json.load(open(ALIGN_PATH))
        # tree instances with realized word and its order in the sentence
        new_instances = []
        # lexicon for the language
        self.lexicon = {}

        for instance in instances:
            tree, sent_id, text = instance['tree'], instance['sent_id'], instance['text']

            if language == 'pt':
                text = contractions.parse(text)

            self.align(tree, text)

            tree['nodes'] = self.nodes
            new_instances.append({ 'tree':tree, 'text': self.tokens, 'sent_id': sent_id })

        json.dump(new_instances, open(ALIGN_PATH, 'w'))
        json.dump(self.lexicon, open(LEXICON_PATH, 'w'))

    def __add_lexicon__(self, node, realization):
        '''
        Method to add the proper realization in the lexicon
        :param node: target node
        :param realization: realization word
        :return:
        '''
        upos = self.nodes[node]['upos']
        if upos not in self.lexicon:
            self.lexicon[upos] = {}

        lemma = self.nodes[node]['lemma']
        if lemma not in self.lexicon[upos]:
            self.lexicon[upos][lemma] = []

        features = copy.copy(self.nodes[node]['feats'])
        features['realization'] = realization

        if features not in map(lambda x: x['features'], self.lexicon[upos][lemma]):
            self.lexicon[upos][lemma].append({'features':features, 'count':1})
        else:
            for i, instance in enumerate(self.lexicon[upos][lemma]):
                if features == instance['features']:
                    self.lexicon[upos][lemma][i]['count'] += 1

    def __match_word__(self):
        '''
        Match node lemma with words in text
        :return:
        '''
        for node in self.nodes:
            lemma = self.nodes[node]['lemma']

            # get the position of the nodes already solved
            order_ids = map(lambda node: self.nodes[node]['order_id'], self.solved)
            f = filter(lambda x: unicode(x) == lemma and x.i not in order_ids, self.doc)
            if len(f) == 1:
                order_id, realization = f[0].i, unicode(f[0])
                self.nodes[node]['order_id'] = order_id
                self.nodes[node]['realization'] = realization #+ u'_word'

                # add in lexicon
                self.__add_lexicon__(node, realization)

                self.solved.append(node)

    def __match_lemma__(self):
        '''
        Match node lemma with lemmas in text
        :return:
        '''
        fnodes = filter(lambda node: node not in self.solved, self.nodes)
        for node in fnodes:
            lemma = self.nodes[node]['lemma']

            # get the position of the nodes already solved
            order_ids = map(lambda node: self.nodes[node]['order_id'], self.solved)
            f = filter(lambda x: x.lemma_ == lemma and x.i not in order_ids, self.doc)

            if len(f) == 1:
                order_id, realization = f[0].i, unicode(f[0])
                self.nodes[node]['order_id'] = order_id
                self.nodes[node]['realization'] = realization #+ u'_lemma'

                # add in lexicon
                self.__add_lexicon__(node, realization)

                self.solved.append(node)

    def __match_dependency__(self, root):
        '''
        Match node lemma with lemmas in text from a given head dep
        :return:
        '''
        if self.root != root and root not in self.solved:
            lemma = self.nodes[root]['lemma']
            upos = self.nodes[root]['upos']
            head_lemma = self.nodes[self.nodes[root]['head']]['lemma']

            # get the position of the nodes already solved
            order_ids = map(lambda node: self.nodes[node]['order_id'], self.solved)
            # filter children of the same head and same lemma of root
            f = filter(lambda token: unicode(token.head) == head_lemma
                                     and token.i not in order_ids
                                     and token.pos_ == upos
                                     and token.lemma_ == lemma, self.doc)

            if len(f) == 1:
                order_id, realization = f[0].i, unicode(f[0])
                self.nodes[root]['order_id'] = order_id
                self.nodes[root]['realization'] = realization #+ u'_dep'

                # add in lexicon
                self.__add_lexicon__(root, realization)

                self.solved.append(root)

        for edge in self.edges[root]:
            self.__match_dependency__(edge['node'])

    def __match_distance__(self):
        '''
        Match node lemma with lemmas in text of shortest string distance
        :return:
        '''
        fnodes = filter(lambda node: node not in self.solved, self.nodes)
        for node in fnodes:
            lemma = self.nodes[node]['lemma']

            # get the position of the nodes already solved
            order_ids = map(lambda node: self.nodes[node]['order_id'], self.solved)
            candidates = filter(lambda x: x.i not in order_ids, self.doc)
            candidates = map(lambda x: (x, edit_distance(x.lemma_, lemma)), candidates)
            candidates.sort(key=lambda x: x[1])

            if len(candidates) > 0:
                order_id, realization = candidates[0][0].i, unicode(candidates[0][0])
                self.nodes[node]['order_id'] = order_id
                self.nodes[node]['realization'] = realization #+ u'_dist'

                # add in lexicon
                # self.__add_lexicon__(node, realization)
            else:
                self.nodes[node]['order_id'] = -1
                self.nodes[node]['realization'] = lemma #+ u'_dist'

            self.solved.append(node)

    def align(self, tree, text):
        '''
        Match a source lemma with the target text words in 4 trials
        :param tree:
        :param text:
        :return:
        '''
        self.doc = self.nlp(text)
        self.tokens = ' '.join(map(lambda token: unicode(token), self.doc))
        self.root, self.nodes, self.edges = tree['root'], tree['nodes'], tree['edges']

        # nodes with order and realization solved
        self.solved = []

        self.__match_word__()

        if len(self.solved) != len(self.nodes):
            self.__match_lemma__()

        if len(self.solved) != len(self.nodes):
            self.__match_dependency__(self.root)

        if len(self.solved) != len(self.nodes):
            self.__match_distance__()

if __name__ == '__main__':
    TRAIN_ALIGN_PATH = 'data/json/train'
    DEV_ALIGN_PATH = 'data/json/dev'

    LEXICON_PATH = 'data/lexicon'
    TRAIN_LEXICON_PATH = 'data/lexicon/train'
    DEV_LEXICON_PATH = 'data/lexicon/dev'
    if not os.path.exists(LEXICON_PATH):
        os.mkdir(LEXICON_PATH)
        os.mkdir(TRAIN_LEXICON_PATH)
        os.mkdir(DEV_LEXICON_PATH)


    for fname in os.listdir(TRAIN_ALIGN_PATH):
        print os.path.join(TRAIN_ALIGN_PATH, fname)
        language = fname.replace('.json', '')
        aligner = Aligner(os.path.join(TRAIN_ALIGN_PATH, fname), os.path.join(TRAIN_LEXICON_PATH, fname), language)

    for fname in os.listdir(DEV_ALIGN_PATH):
        print os.path.join(DEV_ALIGN_PATH, fname)
        language = fname.replace('.json', '')
        aligner = Aligner(os.path.join(DEV_ALIGN_PATH, fname), os.path.join(DEV_LEXICON_PATH, fname), language)