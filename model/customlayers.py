"""
A utility file for custom layers.
"""
import lasagne
import theano
import theano.tensor as T
import numpy as np

class InnerProductLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(InnerProductLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        M = inputs[0]
        u = inputs[1]
        # Takes two 3d tensors of (dim1, dim2, dim3) and outputs tensor of (dim1) representing dim1 dot products of the remaining two axes
        output = T.batched_dot(M, u)
        if self.nonlinearity is not None:
            output = self.nonlinearity(output)
        return output


class BatchedDotLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])


class SumLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axis, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis+1:]

    def get_output_for(self, input, **kwargs):
        return T.sum(input, axis=self.axis)

class ScaleSumLayer(SumLayer):

    def __init__(self, incoming, axis, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis+1:]

    def get_output_for(self, input, **kwargs):
        return T.mean(input, axis=self.axis)

class GramOverlapLayer():

    def __init__(self, incoming, **kwargs):
        super(GramOverlapLayer, self).__init__(incoming, **kwargs)
        
    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input



# An alternate way to represent  sentences into a vector (m_i in diagram)
class TemporalEncodingLayer(lasagne.layers.Layer):  
    """
    TODO: Force self.T to have width query_len, to apply temporal locality only to adjacent premise-hypothesis pairs.
    """
    def __init__(self, incoming, T=lasagne.init.GlorotNormal(), **kwargs):
        super(TemporalEncodingLayer, self).__init__(incoming, **kwargs)
        self.T = self.add_param(T, self.input_shape[-2:], name="T")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input + self.T

# To ensure that W^T (final linear transform) is same as dimension of A, B, C -- will not be relevant for our purposes
# because W will have dim (d x 3) instead of (d x V)
class TransposedDenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class PositionalEncodingLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, vocab, L, **kwargs):
        # Init expects:
        #   l_context_in-- dim (batch_size * max_seq_len * max_sen_len)
        #   l_context_pe_in-- dim (batch_size, max_seq_len, max_sen_len, embed_size)
        #   vocab--
        #   L-- an initialization for the embedding matrix, or None if to use the default initializer.
        super(PositionalEncodingLayer, self).__init__(incomings, **kwargs)
        
        self.vocab = vocab
        if len(incomings) != 2:
            raise NotImplementedError

        self.l_context_in, self.l_context_pe_in = tuple(incomings)
        batch_size, max_seq_len, max_sent_len, embed_size = self.l_context_pe_in.shape
        # This step is important: embedding matrix must be a parameter of this class. 
        self.W = self.add_param(L, (len(vocab) + 1, embed_size), name="W")
        self.embedding = lasagne.layers.EmbeddingLayer(self.l_context_in, len(vocab) + 1, embed_size, W=self.W)

        l_embedding = lasagne.layers.ReshapeLayer(self.embedding, shape=(batch_size, max_seq_len, max_sent_len, embed_size))
        # m_i = sum over j of (l_j * Ax_{ij})
        l_merged = lasagne.layers.ElemwiseMergeLayer((l_embedding, self.l_context_pe_in), merge_function=T.mul)
        self.layer = SumLayer(l_merged, axis=2)
        #---------------------------------------------------

    def get_output_shape_for(self, input_shapes):
        # Output dim: (batch_size, max_seq_len, embed_size)
        _, l_context_pe_shape = tuple(input_shapes)
        return (l_context_pe_shape[0], l_context_pe_shape[1], l_context_pe_shape[3])

    def get_output_for(self, inputs, **kwargs):
        return lasagne.layers.helper.get_output(self.layer, {self.l_context_in: inputs[0], self.l_context_pe_in: inputs[1]})



class SentenceEmbeddingLayer(lasagne.layers.MergeLayer):
    """
    Computes a sentence embedding on the fly using an LSTM. 

    Inputs:
    - incomings: 
      [0]: a tensor of (batch_size, max_seq_len) indicating
           sentence indices
      [1]: a sentence embedding layer that can take (batch_size, max_seq_len)
           to a sentence embedding of size (batch_size, max_seq_len, embedding_size)
           
           For example, this could be a pre-trained portion of a pair LSTM model.

   A_sent: A shared sentence-embedding tensor size (|S|, embedding_size)

    """

    def __init__(self, incomings, A_sent, **kwargs):
        super(SentenceEmbeddingLayer, self).__init__(incoming)

        if len(incomings != 2):
            raise NotImplementedError
        pass




class MemoryNetworkLayer(lasagne.layers.MergeLayer):
    """
    Computes a single memory network 'hop' as given in the paper

    End-To-End Memory Networks (2015)
    Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus 
    http://arxiv.org/pdf/1503.08895v5.pdf

    Inputs:
    - incomings: The incoming layer
    - vocab: A set of vocabulary words
    - A, C: Word embedding matrices
    - A_T, C_T: Temporal embedding matrices
    - nonlinearity: a theano / lasagne function to use 
    as part of the attention mechanism.

    """
    def __init__(self, incomings, vocab, embedding_size, 
                 A, A_T, C, C_T, 
                 nonlinearity=lasagne.nonlinearities.softmax, 
                 **kwargs):

        super(MemoryNetworkLayer, self).__init__(incomings)
        if len(incomings) != 3:
            raise NotImplementedError

        batch_size, max_seqlen, max_sentlen = self.input_shapes[0]
        query_len = kwargs.get('query_len', 2)
    
        assert(query_len)
        print max_seqlen
        print query_len
        assert(max_seqlen % query_len == 0)
        #------------------
        # Inputs
        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))    
        # This is expected to be an embedded query vector, called 'u' in the paper.
        l_B_embedding = lasagne.layers.InputLayer(shape=(batch_size, query_len*embedding_size))
        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        #------------------

        #------------------
        # generate the temporal encoding of sentence seque`nces 
        # using per-word positional encoding which is l_context_pe_in
        pe_A_encoding = PositionalEncodingLayer([l_context_in, l_context_pe_in], vocab, A)
        self.A = pe_A_encoding.W

        l_A_embedding = TemporalEncodingLayer(pe_A_encoding, T=A_T)
        self.A_T = l_A_embedding.T
        l_A_embedding = lasagne.layers.ReshapeLayer(l_A_embedding, shape=(batch_size, int(max_seqlen / query_len), query_len * embedding_size))
        
        pe_C_embedding = PositionalEncodingLayer([l_context_in, l_context_pe_in], vocab, C)
        self.C = pe_C_embedding.W

        l_C_embedding = TemporalEncodingLayer(pe_C_embedding, T=C_T)
        self.C_T = l_C_embedding.T
        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size, int(max_seqlen / query_len), query_len * embedding_size))
        #------------------

        # Performs soft attention to get p_i = softmax (u^T * m_i)
        l_prob = InnerProductLayer((l_A_embedding, l_B_embedding), nonlinearity=nonlinearity)

        # gives weighted sum, o = sum (p_i * c_i) from paper
        l_weighted_output = BatchedDotLayer((l_prob, l_C_embedding))

        # skip-combination o + u from paper
        l_sum = lasagne.layers.ElemwiseSumLayer((l_weighted_output, l_B_embedding))

        self.l_context_in = l_context_in
        self.l_B_embedding = l_B_embedding
        self.l_context_pe_in = l_context_pe_in
        self.network = l_sum

        params = lasagne.layers.helper.get_all_params(self.network, trainable=True)
        values = lasagne.layers.helper.get_all_param_values(self.network, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)

        # Add the zero embedding vector for end-of-sequence
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], 
                                        updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A, self.C]])

    def get_output_shape_for(self, input_shapes):
        # Need the output shape to be (batch_size, 2*hidden_size)
        return lasagne.layers.helper.get_output_shape(self.network)

    def get_output_for(self, inputs, **kwargs):
        return lasagne.layers.helper.get_output(self.network, {self.l_context_in: inputs[0], self.l_B_embedding: inputs[1], self.l_context_pe_in: inputs[2]})

    def reset_zero(self):
        self.set_zero(self.zero_vec)

