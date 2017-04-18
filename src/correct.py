from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from model import Model
from data_reader import load_data, DataReader


flags = tf.flags

# data
flags.DEFINE_string('load_model', "./training/2017-04-17 20-56-12/epoch004_4.9581.model", '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')
flags.DEFINE_string('data_dir',   'data',    'data directory')
flags.DEFINE_integer('num_samples', 300, 'how many words to generate')
flags.DEFINE_float('temperature', 1.0, 'sampling temperature')

# model params
flags.DEFINE_float  ('learning_rate',       1.0,  'starting learning rate')
flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15,                             'dimensionality of character embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',         0.5,                            'dropout. 0 = no dropout')

# optimization
flags.DEFINE_integer('num_unroll_steps',    1,   'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size',          1,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_word_length',     65,   'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


def main(_):
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length)

    print('initialized test dataset reader')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = Model(FLAGS, char_vocab, word_vocab, max_word_length)

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model, 'saved at global step', m.global_step.eval())

        ''' training starts here '''
        rnn_state = session.run(m.initial_rnn_state)
        logits = np.ones((word_vocab.size,))

        while True:

            word = input('Enter a word : ')
            if (len(word) > max_word_length):
                print('Invalid word, maximum word size is ' + max_word_length)
                continue

            char_input = np.zeros((1, 1, max_word_length))
            for i,c in enumerate(word):
                char_input[0, 0, i] = char_vocab[c]

            logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                            {m.input: char_input,
                                             m.initial_rnn_state: rnn_state})

            prob = np.exp(logits)
            prob /= np.sum(prob)

            for i in range(5):
                ix = np.argmax(prob)
                print(str(i) + " - " + word_vocab.token(ix) + ' : ' + str(prob[0][0][ix]))
                prob[0][0][ix] = 0.0

if __name__ == "__main__":
    tf.app.run()
