from nlimodel import *


class LSTMModel(NLIModel):
	"""
	Implements a simple LSTM model.
	"""
	def build_network(self):
		""" 
		Builds an LSTM encoder / decoder.
		"""
		print "=" * 80 
		print "BUILDING single-layer LSTM"
		print "=" * 80

		batch_size, max_seqlen, max_sentlen, embedding_size, vocab = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab
		self.hidden_size = 256
		q = T.imatrix()
		y = T.imatrix()

		self.q_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
		self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)

		S_shared = theano.shared(self.S, borrow=True)
		# Indexed in sentences
		qq = S_shared[q.flatten()].reshape((batch_size, max_seqlen, max_sentlen))

		l_question_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))

		L = build_glove_embedding(self.root_dir + "/data/glove/glove.6B.100d.txt", 
								  hidden_size=embedding_size,
								  word_to_idx=self.word_to_idx)

		embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab) + 1, embedding_size, W=L)
		sum_embeddings = ScaleSumLayer(embedding, axis=2)
		# Force embeddings to (batch_size, max_seqlen, embedding_size)
		l_sequence = lasagne.layers.ReshapeLayer(sum_embeddings, shape=(batch_size, max_seqlen, embedding_size))
		print "compiling lstm"
		l_lstm = lasagne.layers.LSTMLayer(
		        l_sequence, self.hidden_size, 
		        grad_clipping=100, gradient_steps=-1, 
		        unroll_scan=True,
		        nonlinearity=T.tanh,
		        only_return_final=False)
		print 'done compiling'
		#l_forward = lasagne.layers.SliceLayer(l_lstm, -1, 1)

		l_pred = lasagne.layers.DenseLayer(l_lstm, num_units=self.num_classes,
		            nonlinearity=lasagne.nonlinearities.softmax)

		probas = lasagne.layers.helper.get_output(l_pred, {l_question_in: qq})
		    

		pred = T.argmax(probas, axis=1)
		cost = T.nnet.categorical_crossentropy(probas, y).sum() #+ reg_cost

		params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
		grads = T.grad(cost, params)

		scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
		updates = lasagne.updates.adam(scaled_grads, params, learning_rate=self.lr)

		givens = {
		    q: self.q_shared,
		    y: self.a_shared,
		}

		self.train_model = theano.function([], cost, givens=givens, updates=updates, on_unused_input='warn')
		self.compute_pred = theano.function([], pred, givens=givens, on_unused_input='ignore')

		zero_vec_tensor = T.vector()
		self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
		self.set_zero = theano.function([zero_vec_tensor], on_unused_input='ignore')

		self.network = l_pred
