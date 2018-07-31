__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Preordering method

PYTHON VERSION: 2.7
DEPENDENCIES:
    cPickle
    NLTK: http://www.nltk.org/
"""

import cPickle as p
import json
import os

import itertools

import nltk
from nltk.classify import MaxentClassifier, accuracy
nltk.config_megam("./megam-64.opt")

class ClassifierTraining():
    def __init__(self, ftrain, fdev, language):
        print(language)
        print('PARSING')
        trainset = json.load(open(ftrain))
        devset = json.load(open(fdev))

        print('EXTRACTING')
        self.train_c1_features, self.train_c2_features = self.extract_features(trainset)
        self.dev_c1_features, self.dev_c2_features = self.extract_features(devset)

        print('TRAINING')
        self.train_c1_features = map(lambda x: (x['features'], x['class']), self.train_c1_features)
        self.clf_step1 = MaxentClassifier.train(self.train_c1_features, 'megam', trace=0, max_iter=1000)

        self.train_c2_features = map(lambda x: (x['features'], x['class']), self.train_c2_features)
        self.clf_sort_step = MaxentClassifier.train(self.train_c2_features, 'megam', trace=0, max_iter=1000)

        print('DUMPING')
        if not os.path.exists('data/models'):
            os.mkdir('data/models')
        p.dump(self.clf_step1, open('data/models/' + language + '_clf_step1.cPickle', 'w'))
        p.dump(self.clf_sort_step, open('data/models/' + language + '_clf_step2.cPickle', 'w'))

        print('EVALUATING')
        self.evaluate()

    def extract_features(self, lngset):
        self.c1_features, self.c2_features = [], []
        for inst in lngset:
            tree = inst['tree']

            self.root, self.nodes, self.edges = tree['root'], tree['nodes'], tree['edges']
            self.__process__(self.root)
        return self.c1_features, self.c2_features

    def __extract_class1__(self, nodes, label):
        def get_feat(feats, key):
            feature = '-'
            if key in feats:
                feature = feats[key]
            return feature

        c1_features = []
        for node in nodes:
            head = self.nodes[node]['head']

            features = {
                'lemma': self.nodes[node]['lemma'],
                'deps': self.nodes[node]['deps'],
                'upos': self.nodes[node]['upos'],
                'head_lemma': self.nodes[head]['lemma'],
                'head_deps': self.nodes[head]['deps'],
                'head_upos':self.nodes[head]['upos']
            }

            c1_features.append({'features':features, 'class':label})

        return c1_features

    def __extract_class2__(self, nodes, label):
        c2_features = []
        combinations = itertools.combinations(nodes, 2)

        for combination in combinations:
            node1 = combination[0]
            node2 = combination[1]
            head = self.nodes[node1]['head']

            if int(self.nodes[node1]['order_id']) < int(self.nodes[node2]['order_id']):
                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node1]['lemma'],
                    'deps1': self.nodes[node1]['deps'],
                    'upos1': self.nodes[node1]['upos'],
                    'lemma2': self.nodes[node2]['lemma'],
                    'deps2': self.nodes[node2]['deps'],
                    'upos2': self.nodes[node2]['upos'],
                    'position':label
                }
                c2_features.append({'features':features, 'class':'before'})

                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node2]['lemma'],
                    'deps1': self.nodes[node2]['deps'],
                    'upos1': self.nodes[node2]['upos'],
                    'lemma2': self.nodes[node1]['lemma'],
                    'deps2': self.nodes[node1]['deps'],
                    'upos2': self.nodes[node1]['upos'],
                    'position':label
                }
                c2_features.append({'features':features, 'class':'after'})
            else:
                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node1]['lemma'],
                    'deps1': self.nodes[node1]['deps'],
                    'upos1': self.nodes[node1]['upos'],
                    'lemma2': self.nodes[node2]['lemma'],
                    'deps2': self.nodes[node2]['deps'],
                    'upos2': self.nodes[node2]['upos'],
                    'position':label
                }
                c2_features.append({'features':features, 'class':'after'})

                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node2]['lemma'],
                    'deps1': self.nodes[node2]['deps'],
                    'upos1': self.nodes[node2]['upos'],
                    'lemma2': self.nodes[node1]['lemma'],
                    'deps2': self.nodes[node1]['deps'],
                    'upos2': self.nodes[node1]['upos'],
                    'position':label
                }
                c2_features.append({'features':features, 'class':'before'})

        return c2_features

    def __process__(self, root):
        before, after = [], []
        for edge in self.edges[root]:
            node = edge['node']
            deps = self.nodes[node]['deps']
            if int(self.nodes[root]['order_id']) == -1 or int(self.nodes[node]['order_id']) == -1 or deps == 'punct':
                pass
            elif int(self.nodes[node]['order_id']) < int(self.nodes[root]['order_id']):
                before.append(node)
            else:
                after.append(node)

            self.__process__(edge['node'])

        c1_features = self.__extract_class1__(before, 'before')
        c2_features = self.__extract_class2__(before, 'before')
        self.c1_features.extend(c1_features)
        self.c2_features.extend(c2_features)

        c1_features = self.__extract_class1__(after, 'after')
        c2_features = self.__extract_class2__(after, 'after')
        self.c1_features.extend(c1_features)
        self.c2_features.extend(c2_features)

    def evaluate(self):
        dev_c1_features = map(lambda x: (x['features'], x['class']), self.dev_c1_features)
        print(language)
        print('One-step: ', accuracy(self.clf_step1, dev_c1_features))

        dev_c2_features = map(lambda x: (x['features'], x['class']), self.dev_c2_features)
        print('Two-step: ', accuracy(self.clf_sort_step, dev_c2_features))
        print(20 * '-')

class Order():
    def __init__(self, clf_step1, clf_sort_step):
        self.clf_step1 = p.load(open(clf_step1))
        self.clf_sort_step = p.load(open(clf_sort_step))

    def process(self, tree):
        self.root, self.nodes, self.edges = tree['root'], tree['nodes'], tree['edges']

        self.linearize(self.root, 1)

        tree['root'], tree['nodes'], tree['edges'] = self.root, self.nodes, self.edges
        return tree

    def __get_feat__(self, feats, key):
        feature = '-'
        if key in feats:
            feature = feats[key]
        return feature

    def linearize(self, root, order_id):
        before, after = [], []

        for edge in self.edges[root]:
            node = edge['node']

            features = {
                'head_lemma': self.nodes[root]['lemma'],
                'head_deps': self.nodes[root]['deps'],
                'head_upos': self.nodes[root]['upos'],
                'lemma': self.nodes[node]['lemma'],
                'deps': self.nodes[node]['deps'],
                'upos': self.nodes[node]['upos']
            }

            label = self.clf_step1.classify(features)

            if label == 'before':
                before.append(node)
            else:
                after.append(node)

        # treat nodes before
        before = self.sort_stepV2(before, 'before')
        for node in before:
            order_id = self.linearize(node, order_id)

        # treat head
        self.nodes[root]['pred_order_id'] = order_id
        order_id += 1

        # treat nodes after
        after = self.sort_stepV2(after, 'after')
        for node in after:
            order_id = self.linearize(node, order_id)

        return order_id


    def sort_step(self, nodes, position):
        if len(nodes) <= 1:
            return nodes

        half = len(nodes) / 2
        group1 = self.sort_step(nodes[:half], position)
        group2 = self.sort_step(nodes[half:], position)

        result = []
        while len(group1) > 0 or len(group2) > 0:
            if len(group1) == 0:
                result.append(group2[0])
                del group2[0]
            elif len(group2) == 0:
                result.append(group1[0])
                del group1[0]
            else:
                node1, node2 = group1[0], group2[0]
                head = self.nodes[node1]['head']

                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node1]['lemma'],
                    'deps1': self.nodes[node1]['deps'],
                    'upos1': self.nodes[node1]['upos'],
                    'lemma2': self.nodes[node2]['lemma'],
                    'deps2': self.nodes[node2]['deps'],
                    'upos2': self.nodes[node2]['upos'],
                    'position':position
                }
                prob_dist = self.clf_sort_step.prob_classify(features)
                labels = {
                    'before': prob_dist.prob('before'),
                    'after': prob_dist.prob('after')
                }

                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node2]['lemma'],
                    'deps1': self.nodes[node2]['deps'],
                    'upos1': self.nodes[node2]['upos'],
                    'lemma2': self.nodes[node1]['lemma'],
                    'deps2': self.nodes[node1]['deps'],
                    'upos2': self.nodes[node1]['upos'],
                    'position':position
                }
                prob_dist = self.clf_sort_step.prob_classify(features)
                labels['before'] *= prob_dist.prob('after')
                labels['after'] *= prob_dist.prob('before')

                label = sorted(labels.keys(), key=lambda y: labels[y], reverse=True)[0]
                if label == 'before':
                    result.append(node1)
                    result.append(node2)
                else:
                    result.append(node2)
                    result.append(node1)
                del group1[0]
                del group2[0]
        return result

    # Mergesort algorithm fixed
    def sort_stepV2(self, nodes, position):
        if len(nodes) <= 1:
            return nodes

        half = len(nodes) / 2
        group1 = self.sort_step(nodes[:half], position)
        group2 = self.sort_step(nodes[half:], position)

        result = []
        i1, i2 = 0, 0
        while i1 < len(group1) or i2 < len(group2):
            if i1 == len(group1):
                result.append(group2[i2])
                i2 += 1
            elif i2 == len(group2):
                result.append(group1[i1])
                i1 += 1
            else:
                node1, node2 = group1[i1], group2[i2]
                head = self.nodes[node1]['head']

                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node1]['lemma'],
                    'deps1': self.nodes[node1]['deps'],
                    'upos1': self.nodes[node1]['upos'],
                    'lemma2': self.nodes[node2]['lemma'],
                    'deps2': self.nodes[node2]['deps'],
                    'upos2': self.nodes[node2]['upos'],
                    'position':position
                }
                prob_dist = self.clf_sort_step.prob_classify(features)
                labels = {
                    'before': prob_dist.prob('before'),
                    'after': prob_dist.prob('after')
                }

                features = {
                    'head_lemma': self.nodes[head]['lemma'],
                    'head_deps': self.nodes[head]['deps'],
                    'head_upos': self.nodes[head]['upos'],
                    'lemma1': self.nodes[node2]['lemma'],
                    'deps1': self.nodes[node2]['deps'],
                    'upos1': self.nodes[node2]['upos'],
                    'lemma2': self.nodes[node1]['lemma'],
                    'deps2': self.nodes[node1]['deps'],
                    'upos2': self.nodes[node1]['upos'],
                    'position':position
                }
                prob_dist = self.clf_sort_step.prob_classify(features)
                labels['before'] *= prob_dist.prob('after')
                labels['after'] *= prob_dist.prob('before')

                label = sorted(labels.keys(), key=lambda y: labels[y], reverse=True)[0]
                if label == 'before':
                    result.append(node1)
                    i1 += 1
                else:
                    result.append(node2)
                    i2 += 1
        return result

if __name__ == '__main__':
    train_path = 'data/json/train'
    dev_path = 'data/json/dev'

    for fname in os.listdir(train_path):
        language = fname.replace('.json', '')
        ClassifierTraining(os.path.join(train_path, fname), os.path.join(dev_path, fname), language)