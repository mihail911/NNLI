#!/usr/bin/python
import os, sys
sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")
if __name__ == '__main__':
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root_dir)

import argparse
from nlimodel import NLIModel
from feedforward import FeedForwardModel
from lstm import LSTMModel
from memn2n import MemoryNetworkModel
from pairlstm import PairLSTMModel

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


_model_dict = {
	'mn'   : lambda args: MemoryNetworkModel(**args.__dict__), 
	'ff'   : lambda args: FeedForwardModel(**args.__dict__),
	'lstm' : lambda args: LSTMModel(**args.__dict__),
    'pairlstm' : lambda args: PairLSTMModel(**args.__dict__)
			}

def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=50, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--opt_alg', type=str, default='adam', help="Learning algorithm")
    parser.add_argument('--num_hops', type=int, default=1, help='Num hops')
    parser.add_argument('--adj_weight_tying', type='bool', default=True, help='Whether to use adjacent weight tying')
    parser.add_argument('--linear_start', type='bool', default=True, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=True, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=25, help='Num epochs')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='l2 regularization')
    parser.add_argument('--query_len', type=int, default=2, help='max Length of network query' )
    parser.add_argument('--root_dir', type=str, default=_root_dir, help='root directory of project')
    parser.add_argument('--model', type=str, default='pairlstm', help='Which model to use (mn, ff, lstm)')
    parser.add_argument('--context_size', type=int, default=10, help='size of context window')
    parser.add_argument('--exp_name', type=str, default='nlimodelzz2', help='Logger file to use')
    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    model_type = args.__dict__['model']
    print "Model type: ", model_type
    model = _model_dict[model_type](args)
    model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
 	main()