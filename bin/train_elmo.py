
import argparse
import glob
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = args.batch_size  # batch size for each GPU
    unroll_steps = args.unroll_steps
    n_gpus = args.gpus

    # number of tokens in training data (this for 1B Word Benchmark)
    # n_train_tokens = 768648884
    # Calculates the number of tokens in training data
    n_train_tokens = 0
    train_files = glob.glob(args.train_prefix)
    for train_file in train_files:
        with open(train_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                n_train_tokens += len(line.split())

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': args.dropout,
    
     'lstm': {
      'cell_clip': 3,
      'dim': args.state_dim,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': args.emb_dim,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': unroll_steps,
     'n_negative_samples_batch': args.neg_samples,
    }

    prefix = args.train_prefix
    valid_prefix = args.valid_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False, shuffle_on_load=True)
    print('Will load validation')
    # validation_data = BidirectionalLMDataset(valid_prefix, vocab, test=True, shuffle_on_load=False)
    # print('Loaded validation')

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, None, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--valid_prefix', help='Prefix for validation files')
    parser.add_argument('--batch_size', help='Batch size for training', type=int, default=128)
    parser.add_argument('--state_dim', help='Dimensions for LSTM state', type=int, default=4096)
    parser.add_argument('--emb_dim', help='Dimensions for embeddings', type=int, default=512)
    parser.add_argument('--unroll_steps', help='Unroll steps for bilstm', type=int, default=20)
    parser.add_argument('--dropout', help='Dropout propability', type=float, default=0.1)
    parser.add_argument('--gpus', help='Number of GPUs to use for training', type=int, default=2)
    parser.add_argument('--neg_samples', help='Dimensions for embeddings', type=int, default=8192)

    args = parser.parse_args()
    main(args)

