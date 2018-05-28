__author__='thiagocastroferreira'

import sys

def save_texts(freference, fsystem, texts):
    fref = open(freference, 'w')
    fsystem = open(fsystem, 'w')

    ref, hyp = [], []

    for y in texts:
        y_real, y_pred, sent_id = y['y_real'], y['y_pred'], y['sent_id']

        fref.write('# sent_id = ' + sent_id.encode('utf-8'))
        fref.write('\n')
        fref.write('# text = ' + y_real.encode('utf-8'))
        fref.write('\n\n')

        fsystem.write('# sent_id = ' + sent_id.encode('utf-8'))
        fsystem.write('\n')
        fsystem.write('# text = ' + y_pred.encode('utf-8'))
        fsystem.write('\n\n')

        real_tokens = y_real.lower().split()
        pred_tokens = y_pred.lower().split()

        ref.append([real_tokens])
        hyp.append(pred_tokens)

    fref.close()
    fsystem.close()

def save_output(ftexts, fsent_id, fout):
    with open(ftexts) as f:
        texts = f.read().decode('utf-8').split('\n')

    with open(fsent_id) as f:
        sent_ids = f.read().decode('utf-8').split('\n')

    assert len(texts) == len(sent_ids)

    with open(fout, 'w') as f:
        for sent_id, text in zip(sent_ids, texts):
            f.write('# sent_id = ' + sent_id.encode('utf-8'))
            f.write('\n')
            f.write('# text = ' + text.encode('utf-8'))
            f.write('\n\n')

if __name__ == '__main__':
    ftexts = sys.argv[1]
    fsent_id = sys.argv[2]
    fout = sys.argv[3]

    print ftexts
    print fsent_id
    print fout

    save_output(ftexts, fsent_id, fout)
