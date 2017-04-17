from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_word_length, eos='+'):

    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab

    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab

    actual_max_word_length = 0

    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        with codecs.open(os.path.join(data_dir, fname + '.txt'), 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.replace('}', '').replace('{', '').replace('|', '')
                line = line.replace('<unk>', ' | ')
                if eos:
                    line = line.replace(eos, '')

                for word in line.split():
                    if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
                        word = word[:max_word_length-2]

                    word_tokens[fname].append(word_vocab.feed(word))

                    char_array = [char_vocab.feed(c) for c in '{' + word + '}']
                    char_tokens[fname].append(char_array)

                    actual_max_word_length = max(actual_max_word_length, len(char_array))

                if eos:
                    word_tokens[fname].append(word_vocab.feed(eos))

                    char_array = [char_vocab.feed(c) for c in '{' + eos + '}']
                    char_tokens[fname].append(char_array)

    assert actual_max_word_length <= max_word_length

    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    char_tensors = {}
    for fname in ('train', 'valid', 'test'):
        assert len(char_tokens[fname]) == len(word_tokens[fname])

        word_tensors[fname] = np.array(word_tokens[fname], dtype=np.int32)
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), actual_max_word_length], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname] [i,:len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, actual_max_word_length

class DataReader:

    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps, char_vocab):
        self.char_vocab = char_vocab
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        self.word_tensor = word_tensor[:reduced_length]
        self.char_tensor = char_tensor[:reduced_length, :]

        self.amount_of_noise = 0.2 / max_word_length
        self.max_word_length = max_word_length
        self.batch_size =  batch_size
        self.num_unroll_steps = num_unroll_steps
    
    def random_position(self, word):
        if np.argwhere(word == self.char_vocab['}']) <= 2:
            return 1
        return np.random.randint(low=1, high=np.argwhere(word == self.char_vocab['}']) - 1)

    def random_char(self):
        return np.random.randint(low=3, high=self.char_vocab.size)

    def replace_random_character(self, word):
        random_char_position = self.random_position(word)
        word[random_char_position] = self.random_char()
        return word
    
    def delete_random_characeter(self, word):
        random_char_position = self.random_position(word)
        word = np.delete(word, random_char_position)
        return word

    def add_random_character(self, word):
        random_char_position = self.random_position(word)
        word = np.insert(word, random_char_position, self.random_char())
        return word
    
    def transpose_random_characters(self, word):
        random_char_position = self.random_position(word)
        if random_char_position + 1 < len(word):
            word[random_char_position + 1], word[random_char_position] = word[random_char_position], word[random_char_position + 1]
        return word

    def corrupt(self, words):
        print(np.random.uniform())
        corrupted_words = words.copy()
        for word in corrupted_words:
            corruption = np.random.uniform()
            if corruption < 0.25:
                corruption *= 4
                if corruption < 0.25:
                    word = self.replace_random_character(word)
                elif corruption < 0.5:
                    word = self.delete_random_characeter(word)
                elif len(word) + 1 < self.max_word_length and corruption < 0.75:
                    word = self.add_random_character(word)
                elif np.argwhere(word == self.char_vocab['}']) > 2 and corruption < 1.0:
                    word = self.transpose_random_characters(word)
        return corrupted_words

    def iter(self):

        ydata = self.word_tensor.copy()
        corrupted_char_tensor = self.corrupt(self.char_tensor)

        x_batches = corrupted_char_tensor.reshape([self.batch_size, -1, self.num_unroll_steps, self.max_word_length])
        y_batches = ydata.reshape([self.batch_size, -1, self.num_unroll_steps])

        x_batches = list(np.transpose(x_batches, axes=(1, 0, 2, 3)))
        y_batches = list(np.transpose(y_batches, axes=(1, 0, 2)))
        self.length = len(y_batches)
        assert len(x_batches) == len(y_batches)

        for x, y in zip(x_batches, y_batches):
            yield x, y

if __name__ == '__main__':

    _, _, wt, ct, _ = load_data('mkroutikov\\data', 65)
    print(wt.keys())

    count = 0
    for x, y in DataReader(wt['valid'], ct['valid'], 20, 35).iter():
        count += 1
        print(x, y)
        if count > 0:
            break
