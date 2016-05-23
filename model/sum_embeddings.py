#!/usr/bin/env python

import numpy as np
import sys, os
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# Hack for command line invocations
if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)

sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")

import lasagne

from lasagne.layers import EmbeddingLayer, InputLayer, DenseLayer, ConcatLayer, get_output
from lasagne.regularization import regularize_layer_params_weighted, l2, l1, regularize_network_params
from model.embeddings import EmbeddingTable
from theano import printing
from util.stats import Stats
from util.utils import getMinibatchesIdx, convertLabelsToMat, generate_data


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 10
# Number of training sequences in each batch
N_BATCH = 8
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 100
DROPOUT = 0.8
NUM_DENSE_UNITS = 3

# Training / Devving / Testing
OP_MODES = {'train', 'dev', 'test'}
MODE = 'train'


class SumEmbeddingLayer(lasagne.layers.Layer):
    """
    Adding an embedding layer in lasagne is as easy as 
    subclassing lasagne.layers.Layer.
    
    The basic methods to be overridden are
    :get_output_shape_for:  The shape of the output given the shape of the input layer.
    :get_output_for: Should provide the composition of Theano / Lasagne functions
    needed to compute this new layer.

    """
    def __init__(self, incoming, keep_prob=1, mask_time=True, seed=None):
        """ 
        Init.  
        :param keep_prob: The probability of keeping each of the embedding vectors
        :param mask_time: Boolean indicating whether to mask individual parameters
        or an entire vector for a time step (analagous to skip-grams)
        :param seed: The seed for the random number generator, if dropout is used.
        """
        super(SumEmbeddingLayer, self).__init__(incoming)
        self.keep_prob = keep_prob
        if keep_prob != 1:
            self.mask_time = mask_time
            
            stream_seed = seed if seed else np.random.randint( (2 << 64))
            self.srng = RandomStreams(seed=stream_seed)
        

    def get_output_shape_for(self, input_shape):
        """
        Return input shape for summed embeddings
        :param input_shape: Expected to be (batch_size, seq_length, embed_dim)
        :return: Expected to be (batch_size, embed_dim)
        """
        return (input_shape[0], input_shape[2])


    def get_output_for(self, input):
        """
        Sum embeddings for given sample across all time steps (seq_length)
        Dropout is applied to each  if applicable
        :param input:  Expected to be (batch_size, seq_length, embed_dim)
        :return: Summed embeddings of dim (batch_size, embed_dim)
        """
        
        if MODE is not 'train':
            out = input.sum(axis=1)
            return out if self.keep_prob == 1 else (1./self.keep_prob * out)
                
        elif self.keep_prob < 1: 
            mask_shape = (input.shape[2],) if self.mask_time else (input.shape[0], input.shape[2])
            
            mask = srng.binomial(shape=mask_shape, p=self.keep_prob)
            return ((1./self.keep_prob) * mask * input).sum(axis=1)

trainData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/snli_1.0_train.jsonl"
trainDataStats = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/train_dataStats.json"
trainLabels = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/train_labels.json"
valData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/snli_1.0_dev.jsonl"
valDataStats = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/dev_dataStats.json"
valLabels = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/dev_labels.json"


def form_mask_input(num_samples, seq_len, max_seq_len, pad_dir):
    mask = np.zeros((num_samples, max_seq_len)).astype(np.float32)
    if pad_dir == 'right':
        mask[:, 0:seq_len] = 1.
    elif pad_dir == 'left':
        mask[:, -seq_len:] = 1.

    return mask


def convert_idx_to_label(idx_array):
    labels = ["entailment", "neutral", "contradiction"]
    predictions = []
    for entry in idx_array:
        predictions.append(labels[entry])

    return predictions


def main(exp_name, embed_data, train_data, train_data_stats, val_data, val_data_stats,
         test_data, test_data_stats, log_path, batch_size, num_epochs,
         unroll_steps, learn_rate, num_dense, dense_dim, penalty, reg_coeff):
    """
    Main run function for training model.
    :param exp_name:
    :param embed_data:
    :param train_data:
    :param train_data_stats:
    :param val_data:
    :param val_data_stats:
    :param test_data:
    :param test_data_stats:
    :param log_path:
    :param batch_size:
    :param num_epochs:
    :param unroll_steps:
    :param learn_rate:
    :param num_dense: Number of dense fully connected layers to add after concatenation layer
    :param dense_dim: Dimension of dense FC layers -- note this only applies if num_dense > 1
    :param penalty: Penalty to use for regularization
    :param reg_weight: Regularization coeff to use for each layer of network; may
                       want to support different coefficient for different layers
    :return:
    """
    # Set random seed for deterministic results
    np.random.seed(0)
    num_ex_to_train = 30

    # Load embedding table
    table = EmbeddingTable(embed_data)
    vocab_size = table.sizeVocab
    dim_embeddings = table.dimEmbeddings
    embeddings_mat = table.embeddings


    train_prem, train_hyp = generate_data(train_data, train_data_stats, "left", "right", table, seq_len=unroll_steps)
    val_prem, val_hyp = generate_data(val_data, val_data_stats, "left", "right", table, seq_len=unroll_steps)
    train_labels = convertLabelsToMat(train_data)
    val_labels = convertLabelsToMat(val_data)

    # To test for overfitting capabilities of model
    if num_ex_to_train > 0:
        val_prem = val_prem[0:num_ex_to_train]
        val_hyp = val_hyp[0:num_ex_to_train]
        val_labels = val_labels[0:num_ex_to_train]

    # Theano expressions for premise/hypothesis inputs to network
    x_p = T.imatrix()
    x_h = T.imatrix()
    target_values = T.fmatrix(name="target_output")


    # Embedding layer for premise
    l_in_prem = InputLayer((batch_size, unroll_steps))
    l_embed_prem = EmbeddingLayer(l_in_prem, input_size=vocab_size,
                        output_size=dim_embeddings, W=embeddings_mat)

    # Embedding layer for hypothesis
    l_in_hyp = InputLayer((batch_size, unroll_steps))
    l_embed_hyp = EmbeddingLayer(l_in_hyp, input_size=vocab_size,
                        output_size=dim_embeddings, W=embeddings_mat)


    # Ensure embedding matrix parameters are not trainable
    l_embed_hyp.params[l_embed_hyp.W].remove('trainable')
    l_embed_prem.params[l_embed_prem.W].remove('trainable')

    l_embed_hyp_sum = SumEmbeddingLayer(l_embed_hyp)
    l_embed_prem_sum = SumEmbeddingLayer(l_embed_prem)

    # Concatenate sentence embeddings for premise and hypothesis
    l_concat = ConcatLayer([l_embed_hyp_sum, l_embed_prem_sum])

    l_in = l_concat
    l_output = l_concat
    # Add 'num_dense' dense layers with tanh
    # top layer is softmax
    if num_dense > 1:
        for n in range(num_dense):
            if n == num_dense-1:
                l_output = DenseLayer(l_in, num_units=NUM_DENSE_UNITS, nonlinearity=lasagne.nonlinearities.softmax)
            else:
                l_in = DenseLayer(l_in, num_units=dense_dim, nonlinearity=lasagne.nonlinearities.tanh)
    else:
        l_output = DenseLayer(l_in, num_units=NUM_DENSE_UNITS, nonlinearity=lasagne.nonlinearities.softmax)

    network_output = get_output(l_output, {l_in_prem: x_p, l_in_hyp: x_h}) # Will have shape (batch_size, 3)
    f_dense_output = theano.function([x_p, x_h], network_output, on_unused_input='warn')

    # Compute cost
    if penalty == "l2":
        p_metric = l2
    elif penalty == "l1":
        p_metric = l1

    layers = lasagne.layers.get_all_layers(l_output)
    layer_dict = {l: reg_coeff for l in layers}
    reg_cost = reg_coeff * regularize_layer_params_weighted(layer_dict, p_metric)
    cost = T.mean(T.nnet.categorical_crossentropy(network_output, target_values).mean()) + reg_cost
    compute_cost = theano.function([x_p, x_h, target_values], cost)

    # Compute accuracy
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=-1), T.argmax(target_values, axis=-1)),
                      dtype=theano.config.floatX)
    compute_accuracy = theano.function([x_p, x_h, target_values], accuracy)

    label_output = T.argmax(network_output, axis=-1)
    predict = theano.function([x_p, x_h], label_output)

    # Define update/train functions
    all_params = lasagne.layers.get_all_params(l_output, trainable=True)
    updates = lasagne.updates.rmsprop(cost, all_params, learn_rate)
    train = theano.function([x_p, x_h, target_values], cost, updates=updates)

    # TODO: Augment embedding layer to allow for masking inputs

    stats = Stats(exp_name)
    acc_num = 10

    #minibatches = getMinibatchesIdx(val_prem.shape[0], batch_size)
    minibatches = getMinibatchesIdx(train_prem.shape[0], batch_size)
    print("Training ...")
    try:
        total_num_ex = 0
        for epoch in xrange(num_epochs):
            for _, minibatch in minibatches:
                total_num_ex += len(minibatch)
                stats.log("Processed {0} total examples in epoch {1}".format(str(total_num_ex),
                                                                          str(epoch)))

                #prem_batch = val_prem[minibatch]
                #hyp_batch = val_hyp[minibatch]
                #labels_batch = val_labels[minibatch]

                prem_batch = train_prem[minibatch]
                hyp_batch = train_hyp[minibatch]
                labels_batch = train_labels[minibatch]

                train(prem_batch, hyp_batch, labels_batch)
                cost_val = compute_cost(prem_batch, hyp_batch, labels_batch)

                stats.recordCost(total_num_ex, cost_val)
                # Periodically compute and log train/dev accuracy
                if total_num_ex%(acc_num*batch_size) == 0:
                    train_acc = compute_accuracy(train_prem, train_hyp, train_labels)
                    dev_acc = compute_accuracy(val_prem, val_hyp, val_labels)
                    stats.recordAcc(total_num_ex, train_acc, dataset="train")
                    stats.recordAcc(total_num_ex, dev_acc, dataset="dev")

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    embedData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/glove.6B.50d.txt.gz"
    exp_name = "/Users/mihaileric/Documents/Research/LSTM-NLI/log/sum_embeddings.log"
    main(exp_name, embedData, trainData, trainDataStats, valData, valDataStats, "", "",
         "", N_BATCH, NUM_EPOCHS, 18, LEARNING_RATE, num_dense=2, dense_dim=200, penalty="l2", reg_coeff=0.05)
