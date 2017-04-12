import numpy as np
import re
import os
from os import listdir

DEFAULT_DATASET_PATH = './dataset/training-monolingual.tokenized.shuffled'
CLEANED_DATASET_PATH = DEFAULT_DATASET_PATH + '.cleaned'
MAX_DATASET_SIZE = 1000000000

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
        

def parse_dataset(dataset_size = 300000):
    char_vocab = Vocabulary()
    char_vocab.feed(' ')

    word_vocab = Vocabulary()

    max_word_length = 0

    file_names = listdir(CLEANED_DATASET_PATH)

    i_file = 0

    word_tokens = []
    char_tokens = []

    while len(word_tokens) < dataset_size and i_file < len(file_names):

        file = open(CLEANED_DATASET_PATH + '/' + file_names[i_file], mode="r", encoding="utf-8")
        i_file += 1

        for line in file:
            line = line.strip()

            for word in line.split():
                word_tokens.append(word_vocab.feed(word))

                char_array = [char_vocab.feed(c) for c in word]
                char_tokens.append(char_array)

                max_word_length = max(max_word_length, len(char_array))

        file.close()
        print(char_vocab.tokenByIndex_)

def load_dataset(dataset_size = 300000):

    assert(dataset_size < MAX_DATASET_SIZE)

    download_dataset()
    cleanup_dataset() 
    parse_dataset()

if __name__ == "__main__":
    load_dataset()