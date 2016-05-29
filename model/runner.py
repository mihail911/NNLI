#!/usr/bin/python
import os, sys	
if __name__ == '__main__':
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(_root_dir)

import argparse
from nlimodel import NLIModel
from feedforward import FeedForwardModel
from lstm import LSTMModel
from memn2n import MemoryNetworkModel

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--train_file', type=str, default='', help='Train file')
    parser.add_argument('--test_file', type=str, default='', help='Test file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=20, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_hops', type=int, default=3, help='Num hops')
    parser.add_argument('--adj_weight_tying', type='bool', default=True, help='Whether to use adjacent weight tying')
    parser.add_argument('--linear_start', type='bool', default=False, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=False, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='l2 regularization')
    parser.add_argument('--query_len', type=int, default=2, help='max Length of network query' )
    parser.add_argument('--root_dir', type=str, default=_root_dir, help='root directory of project')

    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    model = MemoryNetworkModel(**args.__dict__)
    model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
 	main()