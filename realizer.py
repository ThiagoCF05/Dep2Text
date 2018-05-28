__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Partial realization method

PYTHON VERSION: 2.7
"""

import copy
import json

from order import Order

class Realizer(object):
    def __init__(self, clf_step1, clf_step2, lexicon):
        self.order = Order(clf_step1, clf_step2) #clf_step2_backoff)

        self.lexicon = json.load(open(lexicon))


    def realize(self, root):
        upos = self.nodes[root]['upos']
        lemma = self.nodes[root]['lemma']
        feats = self.nodes[root]['feats']

        self.nodes[root]['prediction'] = lemma
        if lemma in self.lexicon[upos]:
            candidates = self.lexicon[upos][lemma]

            for feat in feats:
                old_candidates = copy.copy(candidates)
                candidates = []
                for candidate in old_candidates:
                    if feat in candidate['features']:
                        if candidate['features'][feat] == feats[feat]:
                            candidates.append(candidate)

                if len(candidates) == 0:
                    candidates = old_candidates

            candidates = sorted(candidates, key=lambda x: x['count'], reverse=True)
            self.nodes[root]['pred_realization'] = candidates[0]['features']['realization']
        else:
            self.nodes[root]['pred_realization'] = lemma


    def linearize(self):
        nodes = sorted(self.nodes, key=lambda node:self.nodes[node]['pred_order_id'])

        text = map(lambda node: self.nodes[node]['pred_realization'], nodes)
        text = ' '.join(text)
        return text


    def remove_punctuation(self):
        punct_nodes = filter(lambda node: self.nodes[node]['deps'] == 'punct' and len(self.edges[node]) == 0, self.nodes)
        for node in punct_nodes:
            head = self.nodes[node]['head']

            new_edges = []
            for edge in self.edges[head]:
                if edge['node'] != node:
                    new_edges.append(edge)
            self.edges[head] = new_edges
            del self.nodes[node]
            del self.edges[node]


    def run(self, tree, remove_punt=True):
        tree = self.order.process(tree)
        self.root, self.nodes, self.edges = tree['root'], tree['nodes'], tree['edges']

        if remove_punt:
            self.remove_punctuation()

        for node in self.nodes:
            self.realize(node)

        text = self.linearize()
        if remove_punt:
            text += ' .'

        tree['root'], tree['nodes'], tree['edges'] = self.root, self.nodes, self.edges
        return tree, text

    def batch_run(self, instances):
        texts, realizations = [], []
        for instance in instances:
            tree, sent_id, text = instance['tree'], instance['sent_id'], instance['text']

            tree, pred_text = self.run(tree)

            texts.append({'y_real':text, 'y_pred':pred_text, 'sent_id': sent_id})

        return texts, realizations