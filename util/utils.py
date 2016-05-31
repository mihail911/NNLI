"""Defines a series of useful utility functions for various modules."""
import cPickle as pickle
import csv
import json
import lasagne
import math
import numpy as np
import os
import re
import sys
import theano

from load_snli_data import loadExampleLabels, loadExampleSentences

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

sick_dir = "../data/SICK/"
data_dir = "../data/SNLI/"

WORD_RE = re.compile(r"([^ \(\)]+)", re.UNICODE)

# Tree parsing 
def str2tree(s):
    """Turns labeled bracketing s into a tree structure (tuple of tuples)"""
    s = WORD_RE.sub(r'"\1",', s)
    s = s.replace(")", "),").strip(",")
    s = s.strip(",")
    return eval(s)

def leaves(t):
    """Returns all of the words (terminal nodes) in tree t"""
    words = []
    for x in t:
        if isinstance(x, str):
            words.append(x)
        else:
            words += leaves(x)
    return words

def tree2sent(t1, t2):
    return ' '.join(leaves(t1)), ' '.join(leaves(t2))


def sick_reader(src_filename):
    count = 1
    print "src file: ", src_filename
    for example in csv.reader(file(src_filename), delimiter="\t"):
        label, t1, t2 = example[:3]
        if not label.startswith('%') and not label=="gold_label": # Some files use leading % for comments.
            yield (label, leaves(str2tree(t1)), leaves(str2tree(t2)))
        count += 1


#Readers for processing SICK datasets
def sick_train_reader():
    return sick_reader(src_filename=sick_dir+"SICK_train_parsed.txt")


def sick_dev_reader():
    return sick_reader(src_filename=sick_dir+"SICK_dev_parsed.txt")


def sick_test_reader():
    return sick_reader(src_filename=sick_dir+"SICK_test_parsed.txt")


def sick_train_dev_reader():
    return sick_reader(src_filename=sick_dir+"SICK_train+dev_parsed.txt")


def snli_reader(src_filename):
    count = 1
    for example in csv.reader(file(src_filename), delimiter="\t"):
        if count == 1:
            count += 1
            continue
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.
            yield (label, leaves(str2tree(t1)), leaves(str2tree(t2)))
        count += 1

#Readers for processing SICK datasets
def snli_train_reader():
    return sick_reader(src_filename=data_dir+"snli_1.0rc3_train.txt")


def snli_dev_reader():
    return sick_reader(src_filename=data_dir+"snli_1.0rc3_dev.txt")


def snli_test_reader():
    return sick_reader(src_filename=data_dir+"snli_1.0rc3_test.txt")

def get_vocab(filenames, reader):
        """ 
        Inputs:
        - filenames: a list of files to look through to form the vocabulary
        - reader: a reader function that takes in the filename and emits a line-by-line tuple 
          of (label, premise, hypothesis) for each ex.

        Returns:
        - vocab: A set of words present in the vocabulary
        - word_to_idx: A bijective word -> idx mapping
        - idx_to_word: A bijective idx -> word mapping
        - max_sentlen: The length of the longest sentence in the dataset
        """
        vocab = set()
        max_sentlen = 0

        for fname in filenames:
            for _, p, h in reader(fname):
                for line in [p, h]:
                    max_sentlen = max(max_sentlen, len(line))
                    for w in line:
                        vocab.add(w)
           
        word_to_idx = {} 
        for w in vocab:
            word_to_idx[w] = len(word_to_idx) + 1

        idx_to_word = {}
        for w, idx in word_to_idx.iteritems():
            idx_to_word[idx] = w

        return vocab, word_to_idx, idx_to_word, max_sentlen


def parse_SICK(filename, word_to_idx):
    """ 
    Parses the SICK dataset into consecutive premise-hypothesis
    sentences.
    - filename: the SICK file to parse
    - word_to_idx: a comprehensive vocabulary mapping to use.
    - 
    """
    labels, sentences = [], []
    for l, premise, hypothesis in sick_reader(filename):
        sentences.append(premise)
        sentences.append(hypothesis)
        labels.append(l)

    # Overfit on smaller dataset
    #return labels[:200], sentences[:400]
    return labels, sentences


def parse_SNLI(filename, word_to_idx):
    """
    Parses the SICK dataset into consecutive premise-hypothesis
    sentences.
    - filename: the SICK file to parse
    - word_to_idx: a comprehensive vocabulary mapping to use.
    -
    """
    labels, sentences = [], []
    for l, premise, hypothesis in snli_reader(filename):
        if l == "-": continue
        sentences.append(premise)
        sentences.append(hypothesis)
        labels.append(l)

    # Overfit on smaller dataset
    #return labels[:200], sentences[:400]
    return labels, sentences



def computeDataStatistics(dataSet="dev"):
    """
    Iterate over data using provided reader and compute statistics. Output values to JSON files.
    :param dataSet: Name of dataset to compute statistics from among 'train', 'dev', or 'test'
    """
    reader = None
    if dataSet == "train":
        reader = snli_train_reader
    elif dataSet == "dev":
        reader = snli_dev_reader
    else:
        reader = snli_test_reader

    # TODO: Fix error in "-" appearing as a label in dataset

    sentences = list()
    labels = list()
    vocab = set()
    minSenLengthPremise = float("inf")
    maxSenLengthPremise = float("-inf")

    minSenLengthHypothesis = float("inf")
    maxSenLengthHypothesis = float("-inf")
    allLabels = ['entailment', 'contradiction', 'neutral']

    for label, t1Tree, t2Tree in reader():
        if label not in allLabels:
            continue
        labels.append(label)

        t1Tokens = leaves(t1Tree)
        t2Tokens = leaves(t2Tree)

        if len(t1Tokens) > maxSenLengthPremise:
            maxSenLengthPremise = len(t1Tokens)
        if len(t1Tokens) < minSenLengthPremise:
            minSenLengthPremise = len(t1Tokens)

        if len(t2Tokens) > maxSenLengthHypothesis:
            maxSenLengthHypothesis = len(t2Tokens)
        if len(t2Tokens) < minSenLengthHypothesis:
            minSenLengthHypothesis = len(t2Tokens)

        vocab.update(set(t1Tokens))
        vocab.update(set(t2Tokens))

        # Append both premise and hypothesis as single list
        # TODO: ensure that sentences cleaned, lower-cased appropriately
        sentences.append([t1Tokens, t2Tokens])

    # Output results to JSON file
    with open(dataSet+"_labels.json", "w") as labelsFile:
        json.dump({'labels': labels}, labelsFile)

    with open(dataSet+"_dataStats.json", "w") as dataStatsFile:
        json.dump({"vocabSize": len(vocab), "minSentLenPremise": minSenLengthPremise,
                   "maxSentLenPremise": maxSenLengthPremise, "minSentLenHypothesis": minSenLengthHypothesis,
                   "maxSentLenHypothesis": maxSenLengthHypothesis}, dataStatsFile)

    with open(dataSet+"_sentences.json", "w") as sentenceFile:
        json.dump({"sentences": sentences}, sentenceFile)


    return vocab, sentences, labels, minSenLengthPremise, maxSenLengthPremise,\
           minSenLengthHypothesis, maxSenLengthHypothesis
    
def build_glove_embedding(filepath, hidden_size, word_to_idx):
    """ 
    Builds a glove vector table for the embeddings from the given filename,
    to the given hidden_size.  If the vectors file has a larger hidden size
    than given, the vectors will be truncated; a hidden size larger than 
    those in the file will be zero-padded.

    The words vectors are mapped into the index dictionary given
    by the third parameter, word_to_idx.
    """

    reader = csv.reader(file(filepath), delimiter=' ', quoting=csv.QUOTE_NONE)
    colnames = None
    print "building glove vectors..."

    mat = 0.05*np.random.randn(len(word_to_idx) + 1, hidden_size)

    for line in reader:        
        if line[0] in word_to_idx: 
            idx = word_to_idx[line[0]]
            mat[idx, :] = np.array(map(float, line[1: ]))
    
    print "Done building vectors"
    return mat

def convertLabelsToMat(dataFile):
    """
    Converts json file of labels to a (numSamples, 3) matrix with a 1 in the column
    corresponding to the label.
    :param dataFile: Path to JSON data file
    :return: numpy matrix corresponding to the labels
    """
    labelsList = ["entailment", "neutral", "contradiction"]
    with open(dataFile, "r") as f:
        labels = loadExampleLabels(dataFile)

        labelsMat = np.zeros((len(labels), 3), dtype=np.float32)
        for idx, label in enumerate(labels):
            labelIdx = labelsList.index(label)
            labelsMat[idx][labelIdx] = 1.

    return labelsMat


def convertMatsToLabel(labelsMat):
    """
    Convert a matrix of labels to a list of labels
    :param labelsMat:
    :return:
    """
    labels = []
    labelsList = ["entailment", "contradiction", "neutral"]
    numSamples, _ = labelsMat.shape
    for idx in range(numSamples):
        sample = labelsMat[idx, :]
        label = labelsList[np.where(sample == 1.)[0][0]]
        labels.append(label)

    return labels


def getMinibatchesIdx(numDataPoints, minibatchSize, shuffle=False):
        """
        Used to shuffle the dataset at each iteration. Return list of
        (batch #, batch) pairs.
        """

        idxList = np.arange(numDataPoints, dtype="int32")

        if shuffle:
            np.random.shuffle(idxList)

        minibatches = []
        minibatchStart = 0
        for i in xrange(numDataPoints // minibatchSize):
            minibatches.append(idxList[minibatchStart:
                                        minibatchStart + minibatchSize])
            minibatchStart += minibatchSize

        if (minibatchStart != numDataPoints):
            # Make a minibatch out of what is left
            minibatches.append(idxList[minibatchStart:])

        return zip(range(len(minibatches)), minibatches)


def convertDataToTrainingBatch(premiseIdxMat, timestepsPremise, hypothesisIdxMat,
                               timestepsHypothesis, pad, embeddingTable, labels, minibatch):
    """
    Convert idxMats to batch tensors for training.
    :param premiseIdxMat:
    :param hypothesisIdxMat:
    :param labels:
    :param pad: Whether zero-padded on left or right
    :return: premise tensor, hypothesis tensor, and batch labels
    """
    if pad == 'right':
        batchPremise = premiseIdxMat[0:timestepsPremise, minibatch, :]
        batchHypothesis = hypothesisIdxMat[0:timestepsHypothesis, minibatch, :]
    else:
        batchPremise = premiseIdxMat[-timestepsPremise:, minibatch, :]
        batchHypothesis = hypothesisIdxMat[-timestepsHypothesis:, minibatch, :]

    batchPremiseTensor = embeddingTable.convertIdxMatToIdxTensor(batchPremise)
    batchHypothesisTensor = embeddingTable.convertIdxMatToIdxTensor(batchHypothesis)

    batchLabels = labels[minibatch]

    return batchPremiseTensor, batchHypothesisTensor, batchLabels


def initShared(value):
    """
    Initialize a shared tensor variable with the given float value
    :param value:
    :return:
    """
    return theano.shared(np.array(value).astype(np.float32))


def computeParamNorms(params, L2regularization):
    """
    Compute the L2 norm of given params with regularization strength
    :param params:
    :param regularization:
    :return:
    """
    rCoef = theano.shared(np.array(L2regularization).astype(np.float32),
                          "regularizationStrength")
    paramSum = 0.
    for param in params:
        paramSum += (param**2).sum()
    paramSum *= rCoef
    return paramSum


def generate_data(data_json_file, data_stats, pad_dir_prem, pad_dir_hyp, embed_table, seq_len):
    """
    Return data of form (num_sample, max_seq_len) where there
    are len in idx list if in masked indices
    :param data_json_file:
    :param data_stats:
    :param pad_dir:
    :param seq_len: desired sequence length
    :return:
    """
    sentences = loadExampleSentences(data_json_file)

    # Not sure if this is completely necessary
    with open(data_stats, "r") as statsFile:
        statsJSON = json.load(statsFile)
        max_len_prem = statsJSON["maxSentLenPremise"]
        max_len_hyp = statsJSON["maxSentLenHypothesis"]

    num_samples = len(sentences)
    prem_mat = np.zeros((num_samples, seq_len)).astype(np.int32)
    hyp_mat = np.zeros((num_samples, seq_len)).astype(np.int32)

    len_vocab = embed_table.sizeVocab

    # Start by filling with idx corresponding to zero embedding vec
    prem_mat.fill(len_vocab-1)
    hyp_mat.fill(len_vocab-1)

    for idx, (premiseSent, hypothesisSent) in enumerate(sentences):
        prem_idx = np.array(embed_table.convertSentListToIdxMatrix(premiseSent))
        hyp_idx = np.array(embed_table.convertSentListToIdxMatrix(hypothesisSent))

        # Pad premise
        if pad_dir_prem == "right":
            prem_mat[idx, 0:len(prem_idx)] = prem_idx[0:seq_len]
        elif pad_dir_prem == "left":
            prem_mat[idx, -len(prem_idx):] = prem_idx[0:seq_len]

        # Pad hypothesis
        if pad_dir_hyp == "right":
            hyp_mat[idx, 0:len(hyp_idx)] = hyp_idx[0:seq_len]
        elif pad_dir_hyp == "left":
            hyp_mat[idx, -len(hyp_idx):] = hyp_idx[0:seq_len]


    return prem_mat, hyp_mat


# TODO: Test this
def saveModel(l_output, file_name):
    """
    Save model parameter values
    :param l_output: Final network output layer
    :return:
    """
    all_params = lasagne.layers.get_all_params(l_output)
    all_param_values = [p.get_value() for p in all_params]

    # Hacky -- first array saved is a single entry array with number of
    # arrays of being saved
    num_params = len(all_param_values)
    with open(file_name, 'w') as f:
        np.save(f, np.array(num_params))
        for param in all_param_values:
            np.save(f, param)


# TODO: Test this!
def loadModel(l_output, file_name):
    """
    Load model from given file name.
    :param file_name:
    :return:
    """
    # Stores all numerical array values
    all_param_values = []
    with open(file_name, 'r') as f:
        num_params = np.load(f)[0]
        for _ in range(num_params):
            all_param_values.append(np.load(f))


    all_params = lasagne.layers.get_all_params(l_output)
    for p, v in zip(all_params, all_param_values):
        p.set_value(v)


# TODO: Put these in a separate initializations util file
def HeKaimingInitializer():
    return lambda shape: np.random.normal(scale=math.sqrt(4.0/(shape[0] + shape[1])), size=shape).astype(np.float32)

def GaussianDefaultInitializer():
    return lambda shape: np.random.randn(shape[0], shape[1]).astype(np.float32)
