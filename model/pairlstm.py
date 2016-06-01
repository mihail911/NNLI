from nlimodel import *


class PairLSTMModel(NLIModel):
	"""
	Implements a simple LSTM model.
	"""
	def build_network(self):
		"""
		Builds an LSTM encoder / decoder.
		"""
		print "=" * 80
		print "BUILDING premise-hypothesis pair LSTM"
		print "=" * 80

		batch_size, max_seqlen, max_sentlen, embedding_size, vocab = self.batch_size, self.max_seqlen, self.max_sentlen, self.embedding_size, self.vocab
		self.hidden_size = 100
		q = T.imatrix()
		y = T.imatrix()

		self.q_shared = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
		self.a_shared = theano.shared(np.zeros((batch_size, self.num_classes), dtype=np.int32), borrow=True)

		S_shared = theano.shared(self.S, borrow=True)
		# Indexed in sentences
		qq = S_shared[q.flatten()].reshape((batch_size, max_seqlen, max_sentlen))

		l_question_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, max_sentlen))






		L = build_glove_embedding(self.root_dir + "/data/glove/glove.6B.50d.txt.gz",
								  hidden_size=embedding_size,
								  word_to_idx=self.word_to_idx)

		embedding = lasagne.layers.EmbeddingLayer(l_question_in, len(vocab) + 1, embedding_size, W=L)
		l_slice_premise = lasagne.layers.SliceLayer(embedding, indices=0, axis=1)

		l_slice_hypothesis = lasagne.layers.SliceLayer(embedding, indices=1, axis=1)



		print "compiling lstm"
		l_lstm_premise = lasagne.layers.LSTMLayer(
		        l_slice_premise, self.hidden_size,
		        grad_clipping=100, gradient_steps=-1,
		        unroll_scan=True,
		        nonlinearity=T.tanh,
		        only_return_final=True)

		l_lstm_hypothesis = lasagne.layers.LSTMLayer(
		        l_slice_hypothesis, self.hidden_size,
		        grad_clipping=100, gradient_steps=-1,
		        unroll_scan=True,
		        nonlinearity=T.tanh,
		        only_return_final=True)
		print 'done compiling'


		l_concat = lasagne.layers.ConcatLayer([l_lstm_premise, l_lstm_hypothesis])

		dense_1 = lasagne.layers.DenseLayer(l_concat, 2*self.hidden_size ,
		                                    W=lasagne.init.GlorotUniform(),
		                                    nonlinearity=T.nnet.relu)
		dense_2 = lasagne.layers.DenseLayer(dense_1, 2*self.hidden_size,
		                                    W=lasagne.init.GlorotUniform(),
		                                    nonlinearity=T.nnet.relu)
		l_pred = lasagne.layers.DenseLayer(dense_2, self.num_classes,
		                                    W=lasagne.init.GlorotNormal(),
		                                    nonlinearity=lasagne.nonlinearities.softmax)


		self.l_concat = l_concat

		probas = lasagne.layers.helper.get_output(l_pred, {l_question_in: qq})


		pred = T.argmax(probas, axis=1)

		reg_coeff = 0.5
		layers = lasagne.layers.get_all_layers(l_pred)
		layer_dict = {l: reg_coeff for l in layers}
		reg_cost = reg_coeff * regularize_layer_params_weighted(layer_dict, l2)

		cost = T.nnet.categorical_crossentropy(probas, y).mean(axis=0, dtype=theano.config.floatX) #+ reg_cost

		params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
		grads = T.grad(cost, params)

		scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)
		updates = lasagne.updates.adam(scaled_grads, params)

		givens = {
		    q: self.q_shared,
		    y: self.a_shared,
		}
		print "Givens..."
		self.train_model = theano.function([], cost, givens=givens, updates=updates, on_unused_input='warn')
		print "Train model compiled"
		self.compute_pred = theano.function([], pred, givens=givens, on_unused_input='ignore')
		print "compute pred compiled"
		zero_vec_tensor = T.vector()
		self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
		self.set_zero = theano.function([zero_vec_tensor], on_unused_input='ignore')

		print "set zero compiled"
		self.network = l_pred

		concat_output = lasagne.layers.helper.get_output(self.l_concat, {l_question_in: qq})
		self.lstm_layer_output = theano.function([], concat_output, givens=givens, on_unused_input='warn')

		print "Done compiling lstm layer function"

