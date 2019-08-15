__author_='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Script for preordering and lexicalizing dependency trees; and making them parallel with gold-standard

PYTHON VERSION: 2.7
"""

import json
import os

from realizer import Realizer

def save_mt_data(texts, fde, fen, fsent):
    fde = open(fde, 'w')
    fen = open(fen, 'w')
    fsent = open(fsent, 'w')

    for text in texts:
        y_real, y_pred, sent_id = text['y_real'], text['y_pred'], text['sent_id']

        fde.write(y_pred.lower().encode('utf-8'))
        fde.write('\n')

        fen.write(y_real.lower().encode('utf-8'))
        fen.write('\n')

        fsent.write(sent_id.encode('utf-8'))
        fsent.write('\n')

    fde.close()
    fen.close()

if __name__ == '__main__':
    TRAIN_PATH = 'data2019/json/train'
    DEV_PATH = 'data2019/json/dev'
    TEST_PATH = 'data2019/json/test'
    MODEL_PATH = 'data2019/models'
    LEXICON_PATH = 'data2019/lexicon/train'
    MT_PATH = 'data2019/mt'

    if not os.path.exists(MT_PATH):
        os.mkdir(MT_PATH)

    for fname in os.listdir(DEV_PATH):
        language = fname.replace('.json', '')
        print language

        LNG_PATH = os.path.join(MT_PATH, language)
        if not os.path.exists(LNG_PATH):
            os.mkdir(LNG_PATH)

        clf_step1 = os.path.join(MODEL_PATH, language + '_clf_step1.cPickle')
        clf_step2 = os.path.join(MODEL_PATH, language + '_clf_step2.cPickle')
        lexicon = os.path.join(LEXICON_PATH, fname)

        realizer = Realizer(clf_step1=clf_step1, clf_step2=clf_step2, lexicon=lexicon)

        traininstances = json.load(open(os.path.join(TRAIN_PATH, fname)))
        texts, realizations = realizer.batch_run(traininstances)
        fde, fen = os.path.join(LNG_PATH, 'train.de'), os.path.join(LNG_PATH, 'train.en')
        fsent_id = os.path.join(LNG_PATH, 'train.sent_id')
        save_mt_data(texts, fde, fen, fsent_id)

        devinstances = json.load(open(os.path.join(DEV_PATH, fname)))
        texts, realizations = realizer.batch_run(devinstances)
        fde, fen = os.path.join(LNG_PATH, 'dev.de'), os.path.join(LNG_PATH, 'dev.en')
        fsent_id = os.path.join(LNG_PATH, 'dev.sent_id')
        save_mt_data(texts, fde, fen, fsent_id)

        # testinstances = json.load(open(os.path.join(TEST_PATH, fname)))
        # texts, realizations = realizer.batch_run(testinstances)
        # fde, fen = os.path.join(LNG_PATH, 'test.de'), os.path.join(LNG_PATH, 'test.en')
        # fsent_id = os.path.join(LNG_PATH, 'test.sent_id')
        # save_mt_data(texts, fde, fen, fsent_id)