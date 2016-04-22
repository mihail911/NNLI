import json
import numpy as np

from util.load_snli_data import loadExampleSentences


class EmbeddingTable(object):
    """
    Reads in data path of word embeddings and
    constructs a matrix of these word embeddings.

    """
    def __init__(self, dataPath, embeddingType="glove"):
        """
            :param dataPath: Path to file with precomputed vectors if relevant
            :param embeddingType: Shorthand for type of embedding
            being built. Will aim to support "glove", "word2vec", and "random"
            options.
        """
        self.type = embeddingType
        self.dataPath = dataPath

        vocab, embeddings, wordToIndex, indexToWord = self._readData(dataPath)

        if vocab != None:
        # Vocabulary of all embeddings contained in table
            self.embeddingVocab = set(vocab)
            self.sizeVocab = embeddings.shape[0]
            self.embeddings = embeddings
            self.dimEmbeddings = embeddings.shape[1]

            # Mapping from word to embedding vector index
            self.wordToIndex = wordToIndex

            # Mapping from embedding vector index to word
            self.indexToWord = indexToWord
        else:
            self.embeddingVocab = None
            self.sizeVocab = None
            self.embeddings = None
            self.dimEmbeddings = 5 # Default size for embeddings if no precomputed embeddings given

            # Mapping from word to embedding vector index
            self.wordToIndex = None

            # Mapping from embedding vector index to word
            self.indexToWord = None


    # Note: There are word vectors for punctuation token -- how to handle those?
    def _readData(self, dataPath):
        """
        Reads in data and constructs a matrix of embeddings, vocabulary,
        and mappings from index:word and word:index
        """
        if dataPath is None:
            return (None, None, None, None)

        wordVocab = []
        wordVectors = []
        with open(dataPath, "r") as f:
            for word in f:
                wordContents = word.split()
                wordVocab.append(wordContents[0])
                wordVectors.append(wordContents[1:])


        # Initialize UNK token to random normally distributed vector with stdDev
        dimVec = len(wordVectors[0])
        stdDev = 0.05
        wordVectors.append(list(stdDev * np.random.randn(dimVec,)))
        wordVocab.append("UNK") # UNK token at end of list of words
        # TODO: Add ZERO token to vocab?? Probably not...

        wordVectors.append([0]*dimVec)
        wordVectors = np.array(wordVectors, dtype=np.float32)
        wordToIndex = dict(zip(wordVocab, range(len(wordVocab))))
        indexToWord = {idx: word for word, idx in wordToIndex.iteritems()}

        return wordVocab, wordVectors, wordToIndex, indexToWord


    def getIdxFromWord(self, word):
        """
        Return the idx in the embedding lookup table for the given word. Return
         last idx of embedding table (associated with UNK token) if word not found in table.
        """
        try:
            idx = self.wordToIndex[word]
            return idx
        except:
            print "Returning index for UNK token..."
            unk_idx = len(self.embeddings) - 2
            return unk_idx


    def getEmbeddingFromWord(self, word):
        """
        Return embedding vector for given word. If embedding not found,
        return a vector of uniformly initialized random values.
        """
        # TODO: Is that proper way to handle unknown words?
        try:
            idx = self.wordToIndex[word]
            return self.embeddings[idx]
        except (KeyError, TypeError):
            print "Word not contained in embedding vocabulary. Returning vector associated with UNK token..."
            unk_idx = len(self.embeddings) - 2
            return self.embeddings[unk_idx]


    def getEmbeddingfromIdx(self, idx):
        """
        Returns word embedding from provided idx, where idx
         assumed to be index of embedding in lookup table.
        :param idx: Idx of word embedding
        """
        try:
            embedding = self.embeddings[idx]
            return embedding
        except:
            # Uniform initialization if embedding idx not found, do this later to train on embeddings
            #return np.random.uniform(-0.05, 0.05, self.dimEmbeddings)
            return np.zeros(self.dimEmbeddings)


    def convertSentToIdxMatrix(self, sentence):
        """
        Converts a given sentence into a matrix of indices for embedding lookup table.
        :param sentence: Sentence to convert into list of idx
        :return: List of idx
        """
        tokens = [word.lower() for word in sentence.split()]
        allIdx = [self.getIdxFromWord(word) for word in tokens]

        return allIdx


    def convertSentListToIdxMatrix(self, sentenceList):
        """
        Converts a given sentence into a matrix of indices for embedding lookup table.
        :param sentence: Sentence to convert into matrix of indices
        :return: List of lists of embedding idx
        """
        tokens = [word.lower() for word in sentenceList]
        allIdx = [self.getIdxFromWord(word) for word in tokens]

        return allIdx


    def convertIdxMatToIdxTensor(self, idxMat):
        """
        Converts an data idxMat of dim (maxSenLength, # samples, 1)
        to (maxSenLength, # samples, dimEmbedding)
        :param idxMat:
        """
        matShape = idxMat.shape
        idxTensor = np.zeros((matShape[0], matShape[1], self.dimEmbeddings),
                             dtype=np.float32)
        for tokenIdx in xrange(matShape[0]):
            for sampleIdx in xrange(matShape[1]):
                embeddingIdx = idxMat[tokenIdx, sampleIdx, 0]
                idxTensor[tokenIdx, sampleIdx, :] = self.getEmbeddingfromIdx\
                                                            (embeddingIdx)
        return idxTensor


    def convertIdxMatrixToEmbeddingList(self, idxList):
        """
        Converts list of idx of form [[2], [5]] where each idx
        represents idx of word embedding to list of appropriate embeddings
        of form [[0.4 0.2], [0.2 0.8]]

        :param idxList: List of word idx
        :return: List of embeddings
        """
        embeddingList = [self.getEmbeddingfromIdx(idx[0]) for idx in idxList]
        return embeddingList


    def convertDataToIdxMatrices(self, dataJSONFile, dataStats, pad='right'):
        """
        Converts data file to matrix of dim (# maxlength, # numSamples, 1)
        where the last dimension stores the idx of the word embedding.
        :param dataJSONFile: File to data with sentences
        :param dataStats:
        :param pad: Whether to pad with zeros at beginning (left) or end (right)
        :return:
        """
        sentences = loadExampleSentences(dataJSONFile)

        with open(dataStats, "r") as statsFile:
            statsJSON = json.load(statsFile)
            maxSentLengthPremise = statsJSON["maxSentLenPremise"]
            maxSentLengthHypothesis = statsJSON["maxSentLenHypothesis"]


        numSent = len(sentences)
        premiseIdxMatrix = np.zeros((maxSentLengthPremise, numSent, 1),
                                    dtype=np.float32)
        hypothesisIdxMatrix = np.zeros((maxSentLengthHypothesis, numSent, 1),
                                       dtype=np.float32)

        # Fill with 'nan' so that we get random embeddings for words that don't exist
        premiseIdxMatrix.fill(np.nan)
        hypothesisIdxMatrix.fill(np.nan)

        for idx, (premiseSent, hypothesisSent) in enumerate(sentences):
            premiseIdxMat = np.array(self.convertSentListToIdxMatrix(premiseSent))[:, 0]
            hypothesisIdxMat = np.array(self.convertSentListToIdxMatrix(hypothesisSent))[:, 0]

            if pad == 'right':
                # Pad with zeros at end
                premiseIdxMatrix[0:len(premiseIdxMat), idx, 0] = premiseIdxMat # Slight bug here
                hypothesisIdxMatrix[0:len(hypothesisIdxMat), idx, 0] = hypothesisIdxMat
            else:
                # Pad with zeros at beginning
                premiseIdxMatrix[-len(premiseIdxMat):, idx, 0] = premiseIdxMat # Slight bug here
                hypothesisIdxMatrix[-len(hypothesisIdxMat):, idx, 0] = hypothesisIdxMat

        return premiseIdxMatrix, hypothesisIdxMatrix


    def convertDataToEmbeddingTensors(self, dataJSONFile, dataStats):
        """
        Reads in JSON file with SNLI sentences and convert to embedding tensor. Note
        sentences below maxLength are padded with zeros.

        :param dataJSONFile: Path to file with SNLI sentences ('train', 'dev', or 'test')
        :param dataStats: Stats about max sent length/vocab size in data file
        :return:
        """
        sentences = loadExampleSentences(dataJSONFile)

        with open(dataStats, "r") as statsFile:
            statsJSON = json.load(statsFile)
            maxSentLengthPremise = statsJSON["maxSentLenPremise"]
            maxSentLengthHypothesis = statsJSON["maxSentLenHypothesis"]


        numSent = len(sentences)
        premiseTensor = np.zeros((maxSentLengthPremise, numSent, self.dimEmbeddings),
                                    dtype=np.float32)
        hypothesisTensor = np.zeros((maxSentLengthHypothesis, numSent,
                                     self.dimEmbeddings), dtype=np.float32)

        for idx, (premiseSent, hypothesisSent) in enumerate(sentences):
            premiseIdxMat = self.convertSentListToIdxMatrix(premiseSent)
            premiseEmbedList = self.convertIdxMatrixToEmbeddingList(premiseIdxMat)
            premiseTensor[0:len(premiseEmbedList), idx, :] = premiseEmbedList # Pad with zeros at end

            hypothesisIdxMat = self.convertSentListToIdxMatrix(hypothesisSent)
            hypothesisEmbedList = self.convertIdxMatrixToEmbeddingList(hypothesisIdxMat)
            hypothesisTensor[0:len(hypothesisEmbedList), idx, :] = hypothesisEmbedList

        return premiseTensor, hypothesisTensor


    def convertIdxMatToSentences(self, idxMat):
        """
        Convert from given idx mat back to collection of sentences that generated
        given idx mat
        :param idxMat:
        :return: list of original sentences
        """
        sentences = []
        numTimeSteps, numSamples, _ = idxMat.shape
        for sampleIdx in range(numSamples):
            sent = []
            for t in range(numTimeSteps):
                embedIdx = idxMat[t, sampleIdx, 0]
                if np.isnan(embedIdx) or embedIdx == -1:
                    continue

                sent.append(self.indexToWord[embedIdx])

            sentences.append(sent)

        return sentences

