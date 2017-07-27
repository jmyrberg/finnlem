# -*- coding: utf8 -*-
"""Training a Dictionary from file(s)."""


import argparse

from dictionary import Dictionary
from model_wrappers import doc_to_tokens
from utils import get_path_files
from data_utils import read_files_batched


parser = argparse.ArgumentParser('Dictionary training')

# Path params (required)
parser.add_argument("--dict-save-path", required=True,
                    type=str, action='store',
                    help='Dictionary save path after training')
parser.add_argument("--dict-train-path", required=True,
                    type=str, action='store',
                    help='Dictionary training data folder or file')

# Dictionary params (optional)
parser.add_argument("--vocab-size", default=30000,
                    type=int, action='store',
                    help='Size of vocabulary')
parser.add_argument("--min-freq", default=0.0,
                    type=float, action='store',
                    help='Minimum token frequency')
parser.add_argument("--max-freq", default=1.0,
                    type=float, action='store',
                    help='Maximum token frequency')

# Training params (optional)
parser.add_argument("--file-batch-size", default=8192,
                    type=int, action='store',
                    help='Dictionary is trained in batches of this size')
parser.add_argument("--prune-every-n", default=200,
                    type=int, action='store',
                    help='Dictionary is pruned every n batch')


def train_dict(args):
    """Fit a Dictionary with training data."""

    # Create a dictionary
    model_dict = Dictionary(vocab_size=args.vocab_size,
                             min_freq=args.min_freq,
                             max_freq=args.max_freq)
    
    # Files to train
    files = get_path_files(args.dict_train_path)

    # Batch generator
    train_gen = read_files_batched(files, 
                                   file_batch_size=args.file_batch_size,
                                   file_batch_shuffle=False,
                                   return_mode='array')
    
    # Fit dictionary in batches
    for docs in train_gen:
        tokens = [doc_to_tokens(doc) for doc in docs.flatten()]
        model_dict.fit_batch(tokens, prune_every_n=args.prune_every_n)
    
    # Save dict
    model_dict.lock()
    model_dict.save(args.dict_save_path)


def main():
    args = parser.parse_args()
    train_dict(args)


if __name__=='__main__':
    main()