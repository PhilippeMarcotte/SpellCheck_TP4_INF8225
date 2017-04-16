import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os
import model
import time
import csv
from Preprocessing import load_dataset, DataReader

import model

TRAINING_DIR = "./training/{}/"

def main(file, batch_size=20, num_unroll_steps=35, char_embed_size=15, rnn_size=650, kernels="[1,2,3,4,5,6,7]", kernel_features="[50,100,150,200,200,200,200]",
         max_grad_norm=5.0, learning_rate=1.0, learning_rate_decay=0.5, decay_when=1.0, seed=3435,
         param_init=0.05, max_epochs=50, print_every=5):
    ''' Trains model from data '''
    directory = TRAINING_DIR.format(time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime()))
    if not os.path.exists(directory):
        os.mkdir(directory)
        print('Created training directory', directory)

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_dataset()

    char_embedding_metadata = os.path.join(directory + "characters_embeddings.tsv")
    with open(char_embedding_metadata, "w", encoding="utf-8") as metadata_file:
        metadata_file.write('padding\n')
        for i in range(1, char_vocab.size()):
            metadata_file.write('%s\n' % (char_vocab.tokenByIndex_[i]))

    train_reader = DataReader(word_tensors['train'], char_tensors['train'],
                              batch_size, num_unroll_steps, char_vocab)

    valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'],
                              batch_size, num_unroll_steps, char_vocab)

    test_reader = DataReader(word_tensors['test'], char_tensors['test'],
                              batch_size, num_unroll_steps, char_vocab)

    print('initialized all dataset readers')

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    with tf.Graph().as_default(), tf.Session(config=config) as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(seed)
        np.random.seed(seed=seed)
        config = projector.ProjectorConfig()
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
                    num_unroll_steps=num_unroll_steps,
                    config=config,
                    char_embedding_metadata=char_embedding_metadata)
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

        summary_writer = tf.summary.FileWriter(directory, graph=session.graph)

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

            save_as = '%s/epoch%03d_%.4f.model' % (directory, epoch, avg_valid_loss)
            saver.save(session, save_as)
            print('Saved model', save_as)

            ''' write out summary events '''
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss),
                tf.Summary.Value(tag="train_perplexity", simple_value=np.exp(avg_train_loss)),
                tf.Summary.Value(tag="valid_perplexity", simple_value=np.exp(avg_valid_loss))])
            summary_writer.add_summary(summary, step)

            projector.visualize_embeddings(summary_writer, config)

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

if __name__ == "__main__":
    tf.app.run()