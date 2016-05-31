from __future__ import division
import os
import argparse
import cPickle as pickle
import glob
import sys
import csv
sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1, regularize_network_params
import nltk
import numpy as np
import random
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from theano.printing import Print as pp
from util.utils import sick_reader, build_glove_embedding, get_vocab, parse_SICK, snli_reader, parse_SNLI
from util.stats import Stats

from customlayers import *
import featurizer

import warnings
warnings.filterwarnings('ignore', '.*topo.*')

_DEBUG=False

class NLIModel:
    """ 
    A model for natural language inference on the SICK or SNLI datasets.  
    Holds the infrastructure for training
    """

    def __init__(self, train_file, test_file, batch_size, 
                embedding_size=20, max_norm=100, lr=0.01, num_hops=3, 
                adj_weight_tying=True, linear_start=False, 
                exp_name='nlirun', context_size=2, **kwargs):

        self.stats = Stats(exp_name)
        self.root_dir = kwargs.get('root_dir')
        la = kwargs.get('opt_alg')

        if la == 'adam':
            self.la = lasagne.updates.adam
        elif la == 'rmsprop':
            self.la = lasagne.updates.rmsprop
        elif la == 'adadelta':
            self.la = lasagne.updates.adadelta
        elif la == 'adagrad':
            self.la = lasagne.updates.adagrad
    

        filenames = (#self.root_dir + "/data/SNLI/snli_1.0rc3_train.txt",
                     #self.root_dir + "/data/SICK/snli_1.0rc3_dev.txt",
                     #self.root_dir + "/data/SICK/SICK_test_parsed.txt")
                    self.root_dir + "/data/SICK/SICK_train_parsed.txt",
                    self.root_dir + "/data/SICK/SICK_dev_parsed.txt",
                    self.root_dir + "/data/SICK/SICK_test_parsed.txt"
                    )
        reader = sick_reader # sick_reader
        vocab, word_to_idx, idx_to_word, max_sentlen = get_vocab(filenames, reader)

        train_labels, train_lines = parse_SICK(filenames[0], word_to_idx)# parse_SICK(filenames[0], word_to_idx)
        test_labels, test_lines = parse_SICK(filenames[1], word_to_idx) # parse_SICK(filenames[1], word_to_idx)
        lines = np.concatenate([train_lines, test_lines], axis=0)

        self.data = {'train': {}, 'test': {}}
        S_train, self.data['train']['C'], self.data['train']['Q'], self.data['train']['Y'], max_seqlen = self.process_dataset(train_lines, word_to_idx, max_sentlen, train_labels, context_size)
        train_len = S_train.shape[0]

        S_test, self.data['test']['C'], self.data['test']['Q'], self.data['test']['Y'], _ = self.process_dataset(test_lines, word_to_idx, max_sentlen,        
                                                                                                      test_labels, context_size, offset=train_len)
        S = np.concatenate([S_train, S_test], axis=0)

        lb = LabelBinarizer()
        lb.fit(train_labels + test_labels)

        print "LB Classes: ", lb.classes_
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        print "-" * 80
        self.query_len = kwargs['query_len']


        self.max_sentlen = max_sentlen
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.num_classes = 3 #len(vocab) + 1
        self.vocab = vocab
        self.adj_weight_tying = adj_weight_tying
        self.num_hops = num_hops
        self.lb = lb
        self.init_lr = lr
        self.lr = self.init_lr
        self.max_norm = max_norm
        self.S = S
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.nonlinearity = None if linear_start else lasagne.nonlinearities.softmax

        self.build_network()
        print "Network built"

    def reset_zero(self): 
        """
        End-to-End memory networks typically set the index 0 embedding to all zeros
        to gracefully ignore the padding at the end / beginning of sequences.

        """
        if not hasattr(self, 'mem_layers'):
            return
        self.set_zero(self.zero_vec)
        for l in self.mem_layers:
            l.reset_zero()

    def predict(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.compute_pred()

    def compute_f1(self, dataset):
        """
        Computes F1 across the dataset and prints a confusion matrix.

        Return:
        - f1_score: The aggreggate class-weighted F1 over the network's predictions
        on the given dataset.
        """
        n_batches = len(dataset['Y']) // self.batch_size
        y_pred = np.concatenate([self.predict(dataset, i) for i in xrange(n_batches)]).astype(np.int32) - 1

        y_true = self.lb.transform([y for y in dataset['Y'][:len(y_pred)]])

        # Convert back to single label representation
        y_true = y_true.argmax(axis=1)
        y_true = y_true.reshape(len(y_pred))
        y_true_confuse = [int(out) - 1 for out in y_true]

        y_true = np.array(y_true_confuse) 

        print metrics.confusion_matrix(y_true, y_pred)
        print metrics.classification_report(y_true, y_pred)
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                errors.append((i, self.lb.classes_[p]))
        return metrics.f1_score(y_true, y_pred, average='weighted', pos_label=None), errors

    def train(self, n_epochs=50, shuffle_batch=True):
        ''' 
        Trains the network for n_epochs, with learning_rate annealing, and 
        verbose printing at each epoch.
        '''
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        self.lr = self.init_lr
        prev_train_f1, prev_test_f1 = None, None


        while (epoch < n_epochs):
            epoch += 1
            # Get the weights from last iter
            last_weights = lasagne.layers.helper.get_all_param_values(self.network)

            # if epoch % 10 == 0:
            #     self.lr /= 2.0

            indices = range(n_train_batches)
            if shuffle_batch:
                self.shuffle_sync(self.data['train'])

            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                total_cost += self.train_model()
                self.reset_zero()
            end_time = time.time()
            print '\n' * 3, '*' * 80
            print 'epoch:', epoch, 'cost:', (total_cost / len(indices)), ' took: %d(s)' % (end_time - start_time)
            self.stats.recordCost(epoch, (total_cost / len(indices)))

            # Subsample train set for computation of accuracy -- pick 1000 random indices
            num_train = len(self.data['train']['Y'])
            subsample_idx = random.sample(np.arange(num_train), 1000)
            subsample_dict = {}
            for key, value in self.data['train'].iteritems():
                subsample_dict[key] = value[subsample_idx]


            ########
            print 'TRAIN', '=' * 40
            train_f1, train_errors = self.compute_f1(subsample_dict)
            self.stats.recordAcc(epoch, train_f1, dataset="train")

            print 'TRAIN_ERROR:', (1-train_f1)*100
            if False:
                for i, pred in train_errors[:10]:
                    print 'context: ', self.to_words(self.data['train']['C'][i])
                    print 'question: ', self.to_words([self.data['train']['Q'][i]])
                    print 'correct answer: ', self.data['train']['Y'][i]
                    print 'predicted answer: ', pred
                    print '---' * 20

            if prev_train_f1 is not None and train_f1 < prev_train_f1 and self.nonlinearity is None:
                # Add nonlinearity back in
                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                self.nonlinearity = lasagne.nonlinearities.softmax
                self.build_network(nonlinearity=lasagne.nonlinearities.softmax)
                lasagne.layers.helper.set_all_param_values(self.network, prev_weights)
            else:
                print 'TEST', '=' * 40
                test_f1, test_errors = self.compute_f1(self.data['test'])
                self.stats.recordAcc(epoch, test_f1, dataset="dev")
                print '*** TEST_ERROR:', (1-test_f1)*100
                # if prev_test_f1 is not None and (test_f1+0.03) < prev_test_f1:
                #     print "REVERTING to last iteration's weights"
                #     lasagne.layers.helper.set_all_param_values(self.network, last_weights)
                #     self.lr /= 2.0
                # else:
                #     pass
                #     #prev_test_f1 = test_f1

            self.save_model("pair_lstm_run1.wts")


        self.stats.recordFinalStats(n_epochs, prev_train_f1, prev_test_f1)

    def to_words(self, indices):
        sents = []
        for idx in indices:
            words = ' '.join([self.idx_to_word[idx] for idx in self.S[idx] if idx > 0])
            sents.append(words)
        return ' '.join(sents)

    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['C', 'Q', 'Y']:
            dataset[k] = dataset[k][p]

    def set_shared_variables(self, dataset, index):
        """
        Sets the shared variables before runnning the network.
        Inputs:
        - dataset: a dictionary with keys 'C', 'Q', 'Y', 
          containing padded context, query, and output sentence indices,
          as necessary.
        """
        q = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['Q'][indices]):
            row = row[:self.max_seqlen]
            q[i, :len(row)] = row

        y[:len(indices), :] = self.lb.transform(dataset['Y'][indices])

        self.q_shared.set_value(q)
        self.a_shared.set_value(y)

    def process_dataset(self, lines, word_to_idx, max_sentlen, labels, context_size, offset=0):
        """ 
        Processes the dataset to emit context, query, and answer vectors
        for an NLIModel.

        Parameters:
        - lines: Iterable of sentences, which are iterables of words
        - word_to_idx: mapping from word to index
        - max_sentlen: The length of the longest sentence
        - labels: Obvious 
        - offset: The desired offset for indexes for query / context sentences.
          If not using a global matrix of sentences across train / dev / test
          sets, keep offset=0.

        Returns:
        - S: A sentence-word index matrix.  Shape (len(lines), max_sentlen)
        - C: A matrix of context sentences per problem.  Shape (len(labels), ??)
        - Q: A matrix of question sentences per problem.  Shape (len(labels), ??)
        - labels: Modified labels, if necessary.
        - max_seqlen: The maximum sequence length in the dataset's context
        """

        S, C, Q, Y = [], [], [], []

        for i, line in enumerate(lines):
            word_indices = [word_to_idx[w] for w in line]
            padding = [0] * (max_sentlen - len(word_indices))
            # front padding for LSTM
            word_indices = padding + word_indices
            S.append(word_indices)
            
            if i % 2: # premise-hypothesis pairs
                C.append([i + offset, i + offset -1])
                Q.append([i + offset, i + offset -1])

        max_seqlen = 2
        return (np.array(S, dtype=np.int32), 
               np.array(C), 
               np.array(Q, dtype=np.int32), 
               np.array(labels), 
                max_seqlen)


    def save_model(self, file_name):
        all_param_values = lasagne.layers.get_all_param_values(self.network)
        #all_param_values = [p.get_value() for p in all_params]
        with open(file_name, "w+") as f:
            pickle.dump(all_param_values, f)


    def load_model(self, file_name):
        with open(file_name, "r") as f:
            params = pickle.load(f)

        lasagne.layers.set_all_param_values(self.network, params)


class MemoryNLIModel(NLIModel):

    """
    A MemoryNLI model loads some memory as a context to aid in the 
    inference task.
    It overloads the following methods:
    - set_shared_variables: to load the dynamic context for each batch
    - process_dataset: for different pre-processing to build contexts
    """

    def set_shared_variables(self, dataset, index):
        """
        Sets the shared variables before runnning the network.
        Inputs:
        - dataset: a dictionary with keys 'C', 'Q', 'Y', 
          containing padded context, query, and output sentence indices,
          as necessary.
        """
        # Commented out for simpler networks

        c = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        q = np.zeros((self.batch_size, self.query_len), dtype=np.int32)
        y = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)
        c_pe = np.zeros((self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)
        q_pe = np.zeros((self.batch_size, self.query_len, self.max_sentlen, self.embedding_size), dtype=theano.config.floatX)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['C'][indices]):
            row = row[:self.max_seqlen]
            c[i, :len(row)] = row

        q[:len(indices)] = dataset['Q'][indices]
        # Create a positional encoding
        for key, mask in [('C', c_pe), ('Q', q_pe)]:
            for i, row in enumerate(dataset[key][indices]):
                sentences = self.S[row].reshape((-1, self.max_sentlen))
                
                for ii, word_idxs in enumerate(sentences):
                    J = np.count_nonzero(word_idxs)
                    for j in np.arange(J):
                        mask[i, ii, j, :] = (1 - (j+1)/J) - ((np.arange(self.embedding_size)+1)/self.embedding_size)*(1 - 2*(j+1)/J)


        y[:len(indices), :] = self.lb.transform(dataset['Y'][indices])

        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)
        self.c_pe_shared.set_value(c_pe)
        self.q_pe_shared.set_value(q_pe)

    def process_dataset(self, lines, word_to_idx, max_sentlen, labels, context_size, offset=0):
        """ 
        Processes the dataset to emit context, query, and answer vectors
        for an NLIModel.

        Parameters:
        - lines: Iterable of sentences, which are iterables of words
        - word_to_idx: mapping from word to index
        - max_sentlen: The length of the longest sentence
        - labels: Obvious 
        - offset: The desired offset for indexes for query / context sentences.
          If not using a global matrix of sentences across train / dev / test
          sets, keep offset=0.

        Returns:
        - S: A sentence-word index matrix.  Shape (len(lines), max_sentlen)
        - C: A matrix of context sentences per problem.  Shape (len(labels), ??)
        - Q: A matrix of question sentences per problem.  Shape (len(labels), ??)
        - labels: Modified labels, if necessary.
        - max_seqlen: The length of the longest context
        """

        S, C, Q, Y = [], [], [], []
        max_seqlen = 2
        for i, line in enumerate(lines):
            word_indices = [word_to_idx[w] for w in line]
            padding = [0] * (max_sentlen - len(word_indices))
            # front padding
            word_indices += padding
            
            S.append(word_indices)
            
            if i % 2: # Premise hypothesis pair
                Q.append([i + offset, i + offset -1])
                # Set context as the last 10 examples
                indices = [i+offset-j for j in np.arange(0, context_size)] # Context size
                max_seqlen = max(max_seqlen, len(indices))
                C.append(indices)

        return (np.array(S, dtype=np.int32), 
               np.array(C), np.array(Q, dtype=np.int32), 
               np.array(labels), max_seqlen)
