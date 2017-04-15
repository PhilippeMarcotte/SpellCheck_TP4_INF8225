import numpy as np
import re
import os
import collections
from os import listdir

DEFAULT_DATASET_PATH = './dataset/training-monolingual.tokenized.shuffled'
CLEANED_DATASET_PATH = DEFAULT_DATASET_PATH + '.cleaned'
MAX_DATASET_SIZE = 1000000000
#NUMBER_OF_LINES = 306068
# TODO: Utiliser comme paramètre d'entrer dans la fonction.
NUMBER_OF_LINES = 30000

NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE) # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768),  chr(769), chr(832),
                                                                                     chr(833),  chr(2387), chr(5151),
                                                                                     chr(5152), chr(65344), chr(8242)),
                                  re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)), re.UNICODE)

class Vocabulary:
    def __init__(self):
        self.indexByToken_ = {}
        self.tokenByIndex_ = []

    def feed(self, token):
        if token not in self.indexByToken_:
            index = len(self.tokenByIndex_)
            self.indexByToken_[token] = index
            self.tokenByIndex_.append(token)
        return self.indexByToken_[token]

    def size(self):
        return len(self.indexByToken_)

def download_dataset():
    # TODO: download dataset here.
    return

def cleanup_line(line):
    """Clean the text - remove unwanted chars, fold punctuation etc."""

    result = NORMALIZE_WHITESPACE_REGEX.sub(' ', line.strip())
    result = RE_DASH_FILTER.sub('-', result)
    result = RE_APOSTROPHE_FILTER.sub("'", result)
    result = RE_LEFT_PARENTH_FILTER.sub("(", result)
    result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
    result = RE_BASIC_CLEANER.sub('', result)
    return result

def cleanup_dataset():
    if os.path.exists(CLEANED_DATASET_PATH):
        return

    os.makedirs(CLEANED_DATASET_PATH)

    file_names = listdir(DEFAULT_DATASET_PATH)

    """Pre-process the data - step 1 - cleanup"""
    with open(CLEANED_DATASET_PATH + '/' + file_names[0], "w", encoding='utf-8') as clean_data:
        for line in open(DEFAULT_DATASET_PATH + '/' + file_names[0], encoding='utf-8'):
            cleaned_line = cleanup_line(line)
            clean_data.write(cleaned_line + "\n")
        

def parse_dataset(training_proportion=0.7, heldout_proportion=0.15):
    char_vocab = Vocabulary()
    char_vocab.feed(' ')

    word_vocab = Vocabulary()

    max_word_length = 0

    file_names = listdir(CLEANED_DATASET_PATH)

    sets = [("train",round(NUMBER_OF_LINES*training_proportion)),("valid",round(NUMBER_OF_LINES*heldout_proportion)),("test",round(NUMBER_OF_LINES*heldout_proportion))]

    word_tokens = collections.defaultdict(list)
    char_tokens = collections.defaultdict(list)

    with open(CLEANED_DATASET_PATH + '/' + file_names[0], mode="r", encoding="utf-8") as file:
            for set_label, set_size in sets:
                for i in range(set_size):
                    for word in file.readline().split():
                        word_tokens[set_label].append(word_vocab.feed(word))

                        char_array = [char_vocab.feed(c) for c in word]
                        char_tokens[set_label].append(char_array)

                        max_word_length = max(max_word_length, len(char_array))

    # on peut supprimer ces prints si tu veux                      
    print()
    print('actual longest token length is:', max_word_length)
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
        char_tensors[fname] = np.zeros([len(char_tokens[fname]), max_word_length], dtype=np.int32)

        for i, char_array in enumerate(char_tokens[fname]):
            char_tensors[fname] [i,:len(char_array)] = char_array

    return word_vocab, char_vocab, word_tensors, char_tensors, max_word_length

def load_dataset(dataset_size = 300000):

    assert(dataset_size < MAX_DATASET_SIZE)

    download_dataset()
    cleanup_dataset() 
    return parse_dataset()
'''
class DataReader:
    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps, char_vocab):
        self.char_vocab = char_vocab
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]

        ydata = word_tensor.copy()

        self.amount_of_noise = 0.2 / max_word_length
        self.max_word_length = max_word_length

        corrupted_char_tensor = self.corrupt(char_tensor)

        x_batches = corrupted_char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)

        self.length = len(self._y_batches)
        self.batch_size =  batch_size
        self.num_unroll_steps = num_unroll_steps
    
    def random_position(self, tensor):
       return np.random.randint(low=0, high=len(tensor))

    def replace_random_character(self, word):
        random_char_position = self.random_position(word)
        random_char_replacement = self.random_position(self.char_vocab.tokenByIndex_)
        word[random_char_position] = random_char_replacement
        return word
    
    def delete_random_characeter(self, word):
        random_char_position = self.random_position(word)
        word = np.delete(word, random_char_position)
        return word

    def add_random_character(self, word):
        random_char_position = self.random_position(word)
        random_char = self.random_position(self.char_vocab.tokenByIndex_)
        word = np.insert(word, random_char_position, random_char)
        return word
    
    def transpose_random_characters(self, word):

        random_char_position = self.random_position(word)
        if random_char_position + 1 < len(word):
            word[random_char_position + 1], word[random_char_position] = word[random_char_position], word[random_char_position + 1]
        return word

    def corrupt(self, words):
        corrupted_words = words.copy()
        for word in corrupted_words:

            if np.random.uniform() < self.amount_of_noise * len(word):
                word = self.replace_random_character(word)
                
            if np.random.uniform() < self.amount_of_noise * len(word):
                word = self.delete_random_characeter(word)

            if len(word) + 1 < self.max_word_length and np.random.uniform() < self.amount_of_noise * len(word):
                word = self.add_random_character(word)

            if np.random.uniform() < self.amount_of_noise * len(word):
                word = self.transpose_random_characters(word)
                
        return corrupted_words

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y
'''

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
    
    def random_position(self, tensor):
       return np.random.randint(low=0, high=len(tensor))

    def replace_random_character(self, word):
        random_char_position = self.random_position(word)
        random_char_replacement = self.random_position(self.char_vocab.tokenByIndex_)
        word[random_char_position] = random_char_replacement
        return word
    
    def delete_random_characeter(self, word):
        random_char_position = self.random_position(word)
        word = np.delete(word, random_char_position)
        return word

    def add_random_character(self, word):
        random_char_position = self.random_position(word)
        random_char = self.random_position(self.char_vocab.tokenByIndex_)
        word = np.insert(word, random_char_position, random_char)
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

            if np.random.uniform() < self.amount_of_noise * len(word):
                word = self.replace_random_character(word)
                
            if np.random.uniform() < self.amount_of_noise * len(word):
                word = self.delete_random_characeter(word)

            if len(word) + 1 < self.max_word_length and np.random.uniform() < self.amount_of_noise * len(word):
                word = self.add_random_character(word)

            if np.random.uniform() < self.amount_of_noise * len(word):
                word = self.transpose_random_characters(word)
                
        return corrupted_words

    def iter(self):

        ydata = self.word_tensor.copy()
        corrupted_char_tensor = self.corrupt(self.char_tensor)

        x_batches = corrupted_char_tensor.reshape([self.batch_size, -1, self.num_unroll_steps, self.max_word_length])
        y_batches = ydata.reshape([self.batch_size, -1, self.num_unroll_steps])

        x_batches = list(np.transpose(x_batches, axes=(1, 0, 2, 3)))
        y_batches = list(np.transpose(y_batches, axes=(1, 0, 2)))

        assert len(x_batches) == len(y_batches)

        for x, y in zip(x_batches, y_batches):
            yield x, y

if __name__ == "__main__":
    load_dataset()