# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 12:13 PM
# @Author  : He Xingwei
'''
This script is used to:
    1. create the vocabulary with training dataset and save it.
    2. load the vocabulary for training.
'''
import pickle as pkl
from collections import Counter
import os
import numpy as np
class Vocab(object):
    def __init__(self, training_files=None, vocab_size=None):
        """
        When creating the vocabulary, training_file and vocab_size should be specified.
        :param training_file:
        :param vocab_size:
        """
        vocabulary_path = os.path.join(os.path.dirname(training_files[0]), 'vocab.pkl')
        if os.path.exists(vocabulary_path):
            print('Loading the vocabulary from {}.'.format(vocabulary_path))
            self.vocabulary_path = vocabulary_path
            self.token_to_id, self.id_to_token = self.load_vocabulary(vocabulary_path)
        else:
            print('Creating the vocabulary with the training_files {}.'.format(training_files))
            if not vocab_size:
                raise ValueError('When creating the vocabulary, you should specify the vocab_size.')
            self.vocabulary_path, self.token_to_id, self.id_to_token = self.make_vocabulary(
                training_files,vocabulary_path,vocab_size)
        # add special tokens
        # special_tokens = ['<UNK>','<PAD>','<BOS>','<EOS>']
        special_tokens = ['<UNK>','<BOS>','<EOS>']
        for token in special_tokens:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token
        self.vocab_size =  len(self.token_to_id)
        self.unk_token_id = self.token_to_id['<UNK>']
        self.bos_token_id = self.token_to_id['<BOS>']
        self.eos_token_id = self.token_to_id['<EOS>']
        print(f'The size of the vocabulary is {self.vocab_size}\n')

        self.bos_token = self.id_to_token.get(self.bos_token_id)
        self.eos_token = self.id_to_token.get(self.eos_token_id)

    def make_vocabulary(self, training_files, vocabulary_path, vocab_size=50000):
        counter = Counter()
        for training_file in training_files:
            with open(training_file) as fr:
                for line in fr:
                    line = line.strip()
                    words = line.split()
                    counter.update(words)
        common_words = counter.most_common()
        _sum = np.sum([value for word, value in common_words])
        common_words = common_words[: vocab_size]
        common_words, values = zip(*common_words)
        print('The vocabulary makes up {:.3f} of the training file.'.format(np.sum(values)/_sum))

        token_to_id = dict(zip(common_words, range(vocab_size)))
        id_to_token = dict(zip(range(vocab_size), common_words))

        with open(vocabulary_path, 'wb') as fw:
            pkl.dump([token_to_id, id_to_token], fw)
        return vocabulary_path, token_to_id, id_to_token

    def load_vocabulary(self, vocabulary_path):
        token_to_id, id_to_token = pkl.load(open(vocabulary_path,'rb'))
        return token_to_id, id_to_token


    def convert_tokens_to_ids(self,tokens):
        """
        tokens must be a list.
        :param tokens:
        :return:
        """
        ids = []
        for value in tokens:
            if type(value) != type([]):
                id = self.token_to_id.get(value, self.unk_token_id)
                ids.append(id)
            else:
                # print(len(value))
                sub_ids = self.convert_tokens_to_ids(value)
                ids.append(sub_ids)
        return ids

    def convert_ids_to_tokens(self, ids):
        """
        ids must be a list.
        :param ids:
        :return:
        """
        tokens = []
        for value in ids:
            if type(value) != type([]):
                assert value < self.vocab_size
                token = self.id_to_token.get(value)
                tokens.append(token)
            else:
                sub_tokens = self.convert_ids_to_tokens(value)
                tokens.append(sub_tokens)
        return tokens
    def encode(self, text,**kwargs):
        """
        convert text to ids.
        :param text:
        :return:
        """
        tokens = text.split()
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def decode(self, ids, **kwargs):
        words = self.convert_ids_to_tokens(ids)
        return ' '.join(words)


if __name__ == "__main__":
    training_file = '../data/one-billion-words/train.txt'
    vocab_size = 50000
    vocab = Vocab(training_files=[training_file],vocab_size=vocab_size)
    # ids = [[50001,50000],50002,[[0],1],[]]
    # print(vocab.convert_ids_to_tokens(ids))
    # ids = []
    # print(vocab.convert_ids_to_tokens(ids))
    s = 'i want to go home now ! but i loss my way .'
    tokens = s.split()
    print(tokens)
    ids = vocab.convert_tokens_to_ids(tokens)
    print(ids)
    tokens = vocab.convert_ids_to_tokens(ids)
    print(tokens)