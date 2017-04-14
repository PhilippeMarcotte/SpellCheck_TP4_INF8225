import tensorflow as tf
import numpy as np
import os
import model
import time
from Preprocessing import load_dataset

import model

TRAINING_DIR = "./training"

class DataReader:

    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps, char_vocab):
        self.char_vocab = char_vocab
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        self.max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        self.word_tensor = word_tensor[:reduced_length]
        self.char_tensor = char_tensor[:reduced_length, :]
        self.amount_of_noise = 0.2/self.max_word_length
        ydata = word_tensor.copy()

        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
    
    def random_position(self, tensor):
       return tf.random_uniform(shape=(1,), minval=0, maxval=len(tensor), dtype=tf.int32)

    def replace_random_character(self, word):
        random_char_position = self.random_position(word)
        random_char_replacement = self.random_position(self.char_vocab.tokenByIndex_)
        word[random_char_position] = tf.gather(self.char_vocab.tokenByIndex_,random_char_replacement)
        return word
    
    def delete_random_characeter(self, word):
        random_char_position = self.random_position(word)
        word = word[:random_char_position] + word[random_char_position + 1:]
        return word

    def add_random_character(self, word):
        random_char_position = self.random_position(word)
        random_char = tf.gather(self.char_vocab,random_position(self.char_vocab.tokenByIndex_))
        word = word[:random_char_position] + random_char + word[random_char_position:]
        return word
    
    def transpose_random_characters(self, word):
        random_char_position = self.random_position(word)
        word = (word[:random_char_position] + word[random_char_position+1] + word[random_char_position] +
                    word[random_char_position + 2:])
        return word

    def corrupt(self, words):
        corrupted_words = words.copy()
        for word in corrupted_words:
            word = tf.cond(tf.random_uniform(shape=(1,)) < self.amount_of_noise * len(word), self.replace_random_character(word), word)

            word = tf.cond(tf.random_uniform(shape=(1,)) < self.amount_of_noise * len(word), self.delete_random_characeter(word), word)

            word = tf.cond(len(word) < self.max_word_length and tf.random_uniform(shape=(1,)) < self.amount_of_noise * len(word), self.add_random_character(word), word)
            
            word = tf.cond(tf.random_uniform(shape=(1,)) < self.amount_of_noise * len(word), self.transpose_random_characters(word), word)
                
        return corrupted_words

    def iter(self):
        corrupted_char_tensor = self.corrupt(self.char_tensor)
        x_batches = self.corrupted_char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = self.ydata.reshape([batch_size, -1, num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)

        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)

        for x, y in zip(x_batches, y_batches):
            yield x, y

def main(file, batch_size=20, num_unroll_steps=35, char_embed_size=15, rnn_size=650, kernels="[1,2,3,4,5,6,7]", kernel_features="[50,100,150,200,200,200,200]",
         max_grad_norm=5.0, learning_rate=1.0, learning_rate_decay=0.5, decay_when=1.0, seed=3435,
         param_init=0.05, max_epochs=25, print_every=5):
    ''' Trains model from data '''

    if not os.path.exists(TRAINING_DIR):
        os.mkdir(TRAINING_DIR)
        print('Created training directory', TRAINING_DIR)

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_dataset()

    train_reader = DataReader(word_tensors['train'], char_tensors['train'],
                              batch_size, num_unroll_steps, char_vocab)

    valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'],
                              batch_size, num_unroll_steps, char_vocab)

    test_reader = DataReader(word_tensors['test'], char_tensors['test'],
                              batch_size, num_unroll_steps, char_vocab)

    print('initialized all dataset readers')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(seed)
        np.random.seed(seed=seed)

        ''' build training graph '''
        initializer = tf.random_uniform_initializer(param_init, param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = model.inference_graph(
                    char_vocab_size=char_vocab.size(),
                    word_vocab_size=word_vocab.size(),
                    char_embed_size=char_embed_size,
                    batch_size=batch_size,
                    rnn_size=rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(kernels),
                    kernel_features=eval(kernel_features),
                    num_unroll_steps=num_unroll_steps)
            train_model.update(model.loss_graph(train_model.logits, batch_size, num_unroll_steps))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(model.training_graph(train_model.loss * num_unroll_steps,
                    learning_rate, max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=50)

        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        with tf.variable_scope("Model", reuse=True):
            valid_model = model.inference_graph(
                    char_vocab_size=char_vocab.size(),
                    word_vocab_size=word_vocab.size(),
                    char_embed_size=char_embed_size,
                    batch_size=batch_size,
                    rnn_size=rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(kernels),
                    kernel_features=eval(kernel_features),
                    num_unroll_steps=num_unroll_steps)
            valid_model.update(model.loss_graph(valid_model.logits, batch_size, num_unroll_steps))

        '''if load_model:
            saver.restore(session, load_model)
            print('Loaded model from', load_model, 'saved at global step', train_model.global_step.eval())
        else:'''
        tf.global_variables_initializer().run()
        session.run(train_model.clear_char_embedding_padding)
        print('Created and initialized fresh model. Size:', model.model_size())

        summary_writer = tf.summary.FileWriter(TRAINING_DIR, graph=session.graph)

        ''' take learning rate from CLI, not from saved graph '''
        session.run(
            tf.assign(train_model.learning_rate, learning_rate),
        )

        ''' training starts here '''
        best_valid_loss = None
        rnn_state = session.run(train_model.initial_rnn_state)
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            avg_train_loss = 0.0
            count = 0
            for x, y in train_reader.iter():
                count += 1
                start_time = time.time()

                loss, _, rnn_state, gradient_norm, step, _ = session.run([
                    train_model.loss,
                    train_model.train_op,
                    train_model.final_rnn_state,
                    train_model.global_norm,
                    train_model.global_step,
                    train_model.clear_char_embedding_padding
                ], {
                    train_model.input  : x,
                    train_model.targets: y,
                    train_model.initial_rnn_state: rnn_state
                })

                avg_train_loss += 0.05 * (loss - avg_train_loss)

                time_elapsed = time.time() - start_time

                if count % print_every == 0:
                    print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                            epoch, count,
                                                            train_reader.length,
                                                            loss, np.exp(loss),
                                                            time_elapsed,
                                                            gradient_norm))

            print('Epoch training time:', time.time()-epoch_start_time)

            # epoch done: time to evaluate
            avg_valid_loss = 0.0
            count = 0
            rnn_state = session.run(valid_model.initial_rnn_state)
            for x, y in valid_reader.iter():
                count += 1
                start_time = time.time()

                loss, rnn_state = session.run([
                    valid_model.loss,
                    valid_model.final_rnn_state
                ], {
                    valid_model.input  : x,
                    valid_model.targets: y,
                    valid_model.initial_rnn_state: rnn_state,
                })

                if count % print_every == 0:
                    print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                avg_valid_loss += loss / valid_reader.length

            print("at the end of epoch:", epoch)
            print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
            print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

            save_as = '%s/epoch%03d_%.4f.model' % (TRAINING_DIR, epoch, avg_valid_loss)
            saver.save(session, save_as)
            print('Saved model', save_as)

            ''' write out summary events '''
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss)
            ])
            summary_writer.add_summary(summary, step)

            ''' decide if need to decay learning rate '''
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(train_model.learning_rate)
                print('learning rate was:', current_learning_rate)
                current_learning_rate *= learning_rate_decay
                if current_learning_rate < 1.e-5:
                    print('learning rate too small - stopping now')
                    break

                session.run(train_model.learning_rate.assign(current_learning_rate))
                print('new learning rate is:', current_learning_rate)
            else:
                best_valid_loss = avg_valid_loss

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()