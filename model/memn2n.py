from nlimodel import *


class MemoryNetworkModel(MemoryNLIModel):
    """
    Provides a 3-hop memory network for NLI with random static
    contexts for a given query.
    """ 

    def build_network(self, nonlinearity=lasagne.nonlinearities.softmax):
        batch_size, max_seqlen, max_sentlen, embedding_size, vocab = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab

        # index form of questions for bAbI
        # {x_i}
        c = T.imatrix()
        # query
        q = T.imatrix()
        # label
        y = T.imatrix()

        # positional encoding of context/query
        c_pe = T.tensor4()
        q_pe = T.tensor4()

        self.c_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        # represents premise/hypothesis pair hence why axis 1 has value 2
        self.q_shared = theano.shared(np.zeros((batch_size, self.query_len), dtype=np.int32), borrow=True)
        self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)
        self.c_pe_shared = theano.shared(np.zeros((batch_size, max_seqlen, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        self.q_pe_shared = theano.shared(np.zeros((batch_size, self.query_len, max_sentlen, embedding_size), dtype=theano.config.floatX), borrow=True)
        
        # the repository of sentences to index into
        S_shared = theano.shared(self.S, borrow=True)
        
        # Actual embeddings of contexts and queries
        # hard-coded 2 to indicate premise/hypothesis pair
        cc = S_shared[c.flatten()].reshape((batch_size, max_seqlen, max_sentlen))
        qq = S_shared[q.flatten()].reshape((batch_size, self.query_len, max_sentlen))

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))
        l_question_in = lasagne.layers.InputLayer(shape=(batch_size, self.query_len, max_sentlen))

        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen, embedding_size))
        # premise/hypothesis pair
        l_question_pe_in = lasagne.layers.InputLayer(shape=(batch_size, self.query_len, max_sentlen, embedding_size))

        A, C = lasagne.init.GlorotNormal().sample((len(vocab)+1, embedding_size)), lasagne.init.GlorotNormal()
        A_T, C_T, B_T = lasagne.init.GlorotNormal(), lasagne.init.GlorotNormal(), lasagne.init.GlorotNormal()
        W = A if self.adj_weight_tying else lasagne.init.GlorotNormal()

        # premise/hypothesis pair
        l_question_in = lasagne.layers.ReshapeLayer(l_question_in, shape=(batch_size * self.query_len * max_sentlen, ))
        # Positional Embedding for b
        l_B_embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab)+1, embedding_size, W=W)
        B = l_B_embedding.W
        # premise/hypothesis pair
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, self.query_len, max_sentlen, embedding_size))
        l_B_embedding = lasagne.layers.ElemwiseMergeLayer((l_B_embedding, l_question_pe_in), merge_function=T.mul)

        #l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, max_sentlen, embedding_size))
        l_B_embedding = SumLayer(l_B_embedding, axis=2)
        l_B_embedding = TemporalEncodingLayer(l_B_embedding, T=B_T)
        self.B_T = l_B_embedding.T
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding, shape=(batch_size, self.query_len*embedding_size))

        self.mem_layers = [MemoryNetworkLayer((l_context_in, l_B_embedding, l_context_pe_in), vocab, embedding_size, 
                                             A=A, A_T=A_T, C=C, C_T=C_T, nonlinearity=nonlinearity, 
                                             **{'query_len' : self.query_len})]
        
        for _ in range(1, self.num_hops):

            if self.adj_weight_tying:
                print "*** RUNNING IN WEIGHT_TIED MODE ***"
                A, C = self.mem_layers[-1].C, lasagne.init.GlorotNormal()
                A_T, C_T = self.mem_layers[-1].C_T, lasagne.init.GlorotNormal()
            else:  # RNN style
                print "*** RUNNING IN RNN MODE *** "
                A, C = self.mem_layers[-1].A, self.mem_layers[-1].C
                A_T, C_T = self.mem_layers[-1].A_T, self.mem_layers[-1].C_T

            self.mem_layers += [MemoryNetworkLayer((l_context_in, self.mem_layers[-1], l_context_pe_in), vocab, embedding_size, 
                                A=A, A_T=A_T, C=C, C_T=C_T, nonlinearity=nonlinearity, 
                                **{'query_len' : self.query_len})]

        l_pred = lasagne.layers.DenseLayer(self.mem_layers[-1], self.num_classes, W=lasagne.init.GlorotNormal(), b=None, nonlinearity=lasagne.nonlinearities.softmax)

        probas = lasagne.layers.helper.get_output(l_pred, {l_context_in: cc, l_question_in: qq, l_context_pe_in: c_pe, l_question_pe_in: q_pe})
        probas = T.clip(probas, 1e-7, 1.0-1e-7)

        pred = T.argmax(probas, axis=1)

        cost = T.nnet.categorical_crossentropy(probas, y).mean(axis=0).sum()

        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
        updates = self.la(grads, params)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared,
            c_pe: self.c_pe_shared,
            q_pe: self.q_pe_shared
        }

        self.train_model = theano.function([], cost, givens=givens, updates=updates, on_unused_input='warn')
        self.compute_pred = theano.function([], pred, givens=givens, on_unused_input='warn')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], 
                                        updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [B]])

        self.nonlinearity = nonlinearity
        self.network = l_pred

