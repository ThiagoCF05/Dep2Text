__author__ = 'thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 25/11/2017
Description:
    NeuralREG+CAtt model concatenating the attention contexts from pre- and pos-contexts

    Based on https://github.com/clab/dynet/blob/master/examples/sequence-to-sequence/attention.py

    Attention()
        :param config
            LSTM_NUM_OF_LAYERS: number of LSTM layers
            EMBEDDINGS_SIZE: embedding dimensions
            STATE_SIZE: dimension of decoding output
            ATTENTION_SIZE: dimension of attention representations
            DROPOUT: dropout probabilities on the encoder and decoder LSTMs
            CHARACTER: character- (True) or word-based decoder
            GENERATION: max output limit
            BEAM_SIZE: beam search size

        train()
            :param fdir
                Directory to save best results and model

    PYTHON VERSION: 3

    DEPENDENCIES:
        Dynet: https://github.com/clab/dynet
        NumPy: http://www.numpy.org/

    UPDATE CONSTANTS:
        FDIR: directory to save results and trained models
"""

import dynet_config
dynet_config.set_gpu()
import dynet as dy
import json
import nltk
import os

from collections import Counter

class Config:
    def __init__(self, config):
        self.lstm_depth = config['LSTM_NUM_OF_LAYERS']
        self.embedding_dim = config['EMBEDDINGS_SIZE']
        self.state_dim = config['STATE_SIZE']
        self.attention_dim = config['ATTENTION_SIZE']
        self.dropout = config['DROPOUT']
        self.max_len = config['GENERATION']
        self.beam = config['BEAM_SIZE']
        self.batch = config['BATCH_SIZE']
        self.early_stop = config['EARLY_STOP']
        self.epochs = config['EPOCHS']

class Preprocess:
    def __init__(self):
        pass

class NeuralREG():
    def __init__(self, config, path):
        self.path = path
        self.config = Config(config=config)

        self.EOS = "eos"
        self.UNK = "unk"
        self.vocab = json.load(open(os.path.join(self.path, 'vocab.json')))
        self.trainset = json.load(open(os.path.join(self.path, 'train.json')))
        self.devset = json.load(open(os.path.join(self.path, 'dev.json')))
        self.testset = json.load(open(os.path.join(self.path, 'test.json')))

        self.vocab = list(self.vocab['input'])
        self.int2word = {i:c for i, c in enumerate(self.vocab['input'])}
        self.word2int = {c:i for i, c in enumerate(self.vocab['input'])}

        self.features_dim = 0

        self.preprocess()
        self.init()


    def preprocess(self):
        vocab = []
        for i, inst in enumerate(self.trainset):
            entity = inst['entity'].replace('_', ' ')
            entity = nltk.word_tokenize(entity)
            vocab.extend(entity)
            self.trainset[i]['entity_parsed'] = entity

        for i, inst in enumerate(self.devset):
            entity = inst['entity'].replace('_', ' ')
            entity = nltk.word_tokenize(entity)
            self.devset[i]['entity_parsed'] = entity

        for i, inst in enumerate(self.testset):
            entity = inst['entity'].replace('_', ' ')
            entity = nltk.word_tokenize(entity)
            self.testset[i]['entity_parsed'] = entity

        vocab.append(self.EOS)
        vocab.append(self.UNK)
        vocab.extend(list(self.vocab['input']))
        vocab = list(set(vocab))

        entities = [w['entity'] for w in self.trainset]
        self.entity_freq = dict([w for w in Counter(entities).items() if w[1] >= 10])


        self.vocab = list(vocab)
        self.int2word = {i:c for i, c in enumerate(vocab)}
        self.word2int = {c:i for i, c in enumerate(vocab)}


    def init(self):
        dy.renew_cg()

        self.VOCAB_SIZE = len(self.vocab)

        self.model = dy.Model()

        # EMBEDDINGS
        self.lookup = self.model.add_lookup_parameters((self.VOCAB_SIZE, self.config.embedding_dim))

        # ENCODERS
        self.enc_fwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim + self.features_dim, self.config.state_dim, self.model)
        self.enc_bwd_lstm = dy.LSTMBuilder(self.config.lstm_depth, self.config.embedding_dim + self.features_dim, self.config.state_dim, self.model)
        self.enc_fwd_lstm.set_dropout(self.config.dropout)
        self.enc_bwd_lstm.set_dropout(self.config.dropout)

        # DECODER
        self.dec_lstm = dy.LSTMBuilder(self.config.lstm_depth, (self.config.state_dim*2)+self.config.embedding_dim, self.config.state_dim, self.model)
        self.dec_lstm.set_dropout(self.config.dropout)

        # ATTENTION
        self.attention_w1 = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * 2))
        self.attention_w2 = self.model.add_parameters((self.config.attention_dim, self.config.state_dim * self.config.lstm_depth * 2))
        self.attention_v = self.model.add_parameters((1, self.config.attention_dim))

        # COPY
        self.copy_x = self.model.add_parameters((1, self.config.embedding_dim + self.features_dim))
        self.copy_decoder = self.model.add_parameters((1, self.config.state_dim * self.config.lstm_depth * 2))
        self.copy_context = self.model.add_parameters((1, self.config.state_dim * 2))
        self.copy_b = self.model.add_parameters((1))

        # SOFTMAX
        self.decoder_w = self.model.add_parameters((self.VOCAB_SIZE, self.config.state_dim))
        self.decoder_b = self.model.add_parameters((self.VOCAB_SIZE))


    def embed_sentence(self, sentence):
        _sentence = list(sentence)
        sentence = []
        for w in _sentence:
            try:
                sentence.append(self.word2int[w])
            except:
                sentence.append(self.word2int[self.UNK])

        return [self.lookup[w] for w in sentence]


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def encode_sentence(self, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors


    def attend(self, encoder_state, decoder_state, w1dt):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attention_w2*dy.concatenate(list(decoder_state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(self.attention_v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = encoder_state * att_weights
        return context, att_weights


    def copy(self, x, decoder_state, context):
        state = dy.concatenate(list(decoder_state.s()))
        return dy.logistic((self.copy_context * context) + (self.copy_decoder * state) + (self.copy_x * x) + self.copy_b)[0]


    def decode(self, encoded, input, output):
        output = [self.word2int[c] for c in output]

        h = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.lookup[self.word2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim*5), last_output_embeddings]))
        loss = []

        for i, wordidx in enumerate(output):
            word = output[i]
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1 * h

            attention, attention_weights = self.attend(h, s, w1dt)

            p_gen = self.copy(last_output_embeddings, s, attention)

            input_prob = dy.scalarInput(0)
            if word in input:
                idx = input.index(word)
                input_prob = dy.pick(attention_weights, idx)

            vocab_prob = dy.scalarInput(0)
            if word in self.vocab:
                vector = dy.concatenate([attention, last_output_embeddings])
                s = s.add_input(vector)
                out_vector = self.decoder_w * s.output() + self.decoder_b
                probs = dy.softmax(out_vector)
                vocab_prob = dy.pick(probs, wordidx)

            try:
                last_output_embeddings = self.lookup[self.word2int[word]]
            except:
                last_output_embeddings = self.lookup[self.word2int[self.UNK]]

            prob = dy.cmult(p_gen, vocab_prob) + dy.cmult(1-p_gen, input_prob)
            loss.append(-dy.log(prob))
        loss = dy.esum(loss)
        return loss


    def generate(self, input):
        embedded = self.embed_sentence(input)
        encoded = self.encode_sentence(embedded)

        h = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.lookup[self.word2int[self.EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.config.state_dim*5), last_output_embeddings]))

        out = []
        count_EOS = 0
        for i in range(self.config.max_len):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or self.attention_w1 * h

            attention, attention_weights = self.attend(h, s, w1dt)

            p_gen = self.copy(last_output_embeddings, s, attention)

            input_probs = dy.cmult(attention_weights, 1-p_gen).vec_value()
            input_prob_max = max(input_probs)
            input_next_word = input_probs.index(input_prob_max)

            vector = dy.concatenate([attention, last_output_embeddings])
            s = s.add_input(vector)
            out_vector = self.decoder_w * s.output() + self.decoder_b
            vocab_probs = dy.cmult(dy.softmax(out_vector), p_gen).vec_value()
            for i, token in enumerate(input):
                if token in self.vocab:
                    vocab_probs[self.word2int[token]] += input_probs[i]
            vocab_prob_max = max(vocab_probs)
            vocab_next_word = vocab_probs.index(vocab_prob_max)

            # If probability of input greater than the vocabulary
            if input_prob_max > vocab_prob_max:
                word = input[input_next_word]
                try:
                    last_output_embeddings = self.lookup[self.word2int[word]]
                except:
                    last_output_embeddings = self.lookup[self.word2int[self.UNK]]
            else:
                last_output_embeddings = self.lookup[vocab_next_word]
                word = self.int2word[vocab_next_word]

            if word == self.EOS:
                count_EOS += 1
                continue

            out.append(word)

        return out


    def get_loss(self, input, output):
        embedded = self.embed_sentence(input)
        encoded = self.encode_sentence(embedded)

        return self.decode(encoded, input, output)


    def write(self, fname, outputs):
        f = open(fname, 'w')
        for output in outputs:
            f.write(output[0])
            f.write('\n')

        f.close()


    def validate(self):
        results = []
        num, dem = 0.0, 0.0
        for i, devinst in enumerate(self.devset):
            pre_context = [self.EOS] + devinst['pre_context']
            pos_context = devinst['pos_context'] + [self.EOS]
            entity = devinst['entity']
            entity_parsed = devinst['entity_parsed']
            # if self.config.beam == 1:
            outputs = [self.generate(pre_context)]
            # else:
            #     outputs = self.beam_search(pre_context, pos_context, entity, self.config.beam)

            delimiter = ' '
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace(self.EOS, '').strip()
            refex = delimiter.join(devinst['refex']).replace(self.EOS, '').strip()

            best_candidate = outputs[0]
            if refex == best_candidate:
                num += 1
            dem += 1

            if i < 20:
                print ("Refex: ", refex, "\t Output: ", best_candidate)
                print(10 * '-')

            results.append(outputs)

            if i % self.config.batch == 0:
                dy.renew_cg()

        return results, num, dem


    def train(self):
        trainer = dy.AdadeltaTrainer(self.model)

        log = []
        best_acc, repeat = 0.0, 0
        for epoch in range(self.config.epochs):
            dy.renew_cg()
            losses = []
            closs = 0.0
            for i, traininst in enumerate(self.trainset):
                pre_context = [self.EOS] + traininst['pre_context']
                pos_context = traininst['pos_context'] + [self.EOS]
                refex = [self.EOS] + traininst['refex'] + [self.EOS]

                loss = self.get_loss(pre_context, refex)
                losses.append(loss)

                if len(losses) == self.config.batch:
                    loss = dy.esum(losses)
                    closs += loss.value()
                    loss.backward()
                    trainer.update()
                    dy.renew_cg()
                    print("Epoch: {0} \t Loss: {1} \t Progress: {2}".format(epoch, round(closs / self.config.batch, 2), round(i / len(self.trainset), 2)), end='       \r')
                    losses = []
                    closs = 0.0

            outputs, num, dem = self.validate()
            acc = float(num) / dem
            log.append(acc)

            print("Dev acc: {0} \t Best acc: {1}".format(round(acc, 2), best_acc))

            # Saving the model with best accuracy
            if best_acc == 0.0 or acc > best_acc:
                best_acc = acc

                fname = 'dev_best_copy.txt'
                self.write(os.path.join(self.path, 'results', fname), outputs)

                fname = 'best_model_copy.dy'
                self.model.save(os.path.join(self.path, 'results', fname))

                repeat = 0
            else:
                repeat += 1

            # In case the accuracy does not increase in EARLY_STOP epochs, break the process
            if repeat == self.config.early_stop:
                break

                # json.dump(log, open(os.path.join(self.path, 'log.json'), 'w'))


    def evaluate(self, procset, write_path):
        results = []
        num, dem = 0.0, 0.0
        for i, devinst in enumerate(procset):
            print('Progress: ', round(i / len(procset), 2), end='\r')
            pre_context = [self.EOS] + devinst['pre_context']
            pos_context = devinst['pos_context'] + [self.EOS]
            entity = devinst['entity']
            entity_parsed = devinst['entity_parsed']
            # if self.config.beam == 1:
            outputs = [self.generate(pre_context)]
            # else:
            #     outputs = self.beam_search(pre_context, pos_context, entity, self.config.beam)

            delimiter = ' '
            for j, output in enumerate(outputs):
                outputs[j] = delimiter.join(output).replace(self.EOS, '').strip()
            refex = delimiter.join(devinst['refex']).replace(self.EOS, '').strip()

            best_candidate = outputs[0]
            if refex == best_candidate:
                num += 1
            dem += 1

            results.append(outputs)

            if i % self.config.batch == 0:
                dy.renew_cg()

        self.write(write_path, results)


    def save(self, path):
        self.model.save(path)


    def populate(self, path):
        self.model.populate(path)


if __name__ == '__main__':
    config = {
        'LSTM_NUM_OF_LAYERS':1,
        'EMBEDDINGS_SIZE':300,
        'STATE_SIZE':512,
        'ATTENTION_SIZE':512,
        'DROPOUT':0.2,
        'GENERATION':30,
        'BEAM_SIZE':1,
        'BATCH_SIZE': 80,
        'EPOCHS': 60,
        'EARLY_STOP': 10
    }

    path = '/roaming/tcastrof/inlg2019'
    neuralreg = NeuralREG(path=path, config=config)
    neuralreg.train()

    path = '/roaming/tcastrof/inlg2019'
    neuralreg = NeuralREG(path=path, config=config)
    neuralreg.populate(os.path.join(path, 'results', 'best_model_copy.dy'))
    neuralreg.evaluate(neuralreg.devset, os.path.join(path, 'results', 'dev_copy.out'))
    neuralreg.evaluate(neuralreg.testset, os.path.join(path, 'results', 'test_copy.out'))
