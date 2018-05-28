__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 28/05/2018
Description:
    Script to clean the Europarl parallel corpus

PYTHON VERSION: 2.7
"""

import os
import sys

def preprocess(fname):
    # print fname
    with open(fname) as f:
        doc = f.read().decode('utf-8').split('\n')

    text = []
    for line in doc:
        # not empty lines
        if len(line) > 0:
            # not xml lines
            if line[0] != '<':
                text.append(line)
    return text

if __name__ == '__main__':
    EUROPARL_DIR = sys.argv[1]
    FOUT = sys.argv[2]

    text = []
    for fname in os.listdir(EUROPARL_DIR):
        text.extend(preprocess(os.path.join(EUROPARL_DIR, fname)))

    with open(FOUT, 'w') as f:
        f.write('\n'.join(text).lower().encode('utf-8'))