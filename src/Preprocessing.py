import numpy as np
import re
import os
import collections
from os import listdir

DEFAULT_DATASET_PATH = './dataset/training-monolingual.tokenized.shuffled'
CLEANED_DATASET_PATH = DEFAULT_DATASET_PATH + '.cleaned'
MAX_DATASET_SIZE = 1000000000
NUMBER_OF_LINES = 306068

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

if __name__ == "__main__":
    load_dataset()