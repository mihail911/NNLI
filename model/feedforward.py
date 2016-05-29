from nlimodel import *

class FeedForwardModel(NLIModel):

	# @Override
	def build_network(self):
		"""
		Builds a very simple feedforward network to assess the effectiveness of training on the task.
		- Input: (batch_size, max_seqlen, max_sentlen)
		- Wordwise embedding into (batch_size, max_seqlen, max_sentlen, embed_size) 
		- Sum all words in a sentence: (batch_size, max_seqlen, embed_size)
		- Reshape embedding into (batch_size, max_seqlen * embed_size)
		- 3 hidden layers with sigmoid, hidden dim (512, 512, 256)
		"""

		batch_size, max_seqlen, max_sentlen, embedding_size, vocab = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab

		self.hidden_size = 256
		q = T.imatrix()
		y = T.imatrix()

		self.q_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
		self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)

		S_shared = theano.shared(self.S, borrow=True)
		qq = S_shared[q.flatten()].reshape((batch_size, max_seqlen, max_sentlen))

		l_question_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))


		L = build_glove_embedding(self.root_dir + "/data/glove/glove.6B.100d.txt", 
		                          embedding_size,
		                          self.word_to_idx)
		print L

		embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab) + 1, embedding_size, W=L)

		sum_embeddings = ScaleSumLayer(embedding, axis=2)

		reshape_sum = lasagne.layers.ReshapeLayer(sum_embeddings, shape=(batch_size, max_seqlen*embedding_size))
		# Fully connected layers

		dense_1 = lasagne.layers.DenseLayer(reshape_sum, self.hidden_size , 
		                                    W=lasagne.init.GlorotNormal(), 
		                                    nonlinearity=T.nnet.sigmoid)
		dense_2 = lasagne.layers.DenseLayer(dense_1, self.hidden_size,  
		                                    W=lasagne.init.GlorotNormal(),
		                                    nonlinearity=T.nnet.sigmoid)
		l_pred = lasagne.layers.DenseLayer(dense_2, self.num_classes, 
		                                    W=lasagne.init.GlorotNormal(),
		                                    nonlinearity=lasagne.nonlinearities.softmax)
		                                    

		rand_in = np.random.randint(0, len(vocab) - 1, size=(batch_size, max_seqlen, max_sentlen))

		fake_probs = lasagne.layers.get_output(l_pred, {l_question_in : rand_in} ).eval()
		print "fake_probs: ", fake_probs  

		probas = lasagne.layers.helper.get_output(l_pred, {l_question_in: qq})

		pred = T.argmax(probas, axis=1)

		# l2 regularization
		reg_coeff = 1e-3
		p_metric = l2
		layer_dict = {dense_1: reg_coeff, dense_2 : reg_coeff, l_pred : reg_coeff}
		reg_cost = reg_coeff * regularize_layer_params_weighted(layer_dict, p_metric)

		cost = T.nnet.categorical_crossentropy(probas, y).sum() + reg_cost

		params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
		grads = T.grad(cost, params)

		scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
		updates = lasagne.updates.adam(scaled_grads, params, learning_rate=self.lr)

		givens = {
		    q: self.q_shared,
		    y: self.a_shared,
		}

		self.train_model = theano.function([], cost, givens=givens, updates=updates, on_unused_input='ignore')
		self.compute_pred = theano.function([], pred, givens=givens, on_unused_input='ignore')

		zero_vec_tensor = T.vector()
		self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
		self.set_zero = theano.function([zero_vec_tensor], on_unused_input='ignore')
		#self.nonlinearity = nonlinearity
		self.network = l_pred
