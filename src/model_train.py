# -*- coding: utf8 -*-
"""Sequence to sequence model training."""


import time
import argparse

from datetime import datetime

import numpy as np

from model_wrappers import Seq2Seq, get_Seq2Seq_model_param_names
from utils import get_path_files
from data_utils import read_file, read_files_cycled, rebatch


parser = argparse.ArgumentParser('Model training')

# Path params (required)
parser.add_argument("--model-dir", required=True,
                    type=str, action='store',
                    help='Model checkpoint and log save path')
parser.add_argument("--train-data-path", required=True,
                    type=str, action='store',
                    help='Training data folder or file')

# Path params (required for the first time)
parser.add_argument("--dict-path", default=None,
                    type=str, action='store',
                    help='Path to existing Dictionary')

# Path params (optional)
parser.add_argument("--validation-data-path", default=None,
                    type=str, action='store',
                    help='Validation data folder or file')

# Model params (required for the first time)
parser.add_argument("--cell-type", default='lstm',
                    type=str, action='store',
                    help="Cell type, either 'gru' or 'lstm'")
parser.add_argument("--hidden-dim", default=32,
                    type=int, action='store',
                    help='Number of neurons in hidden layers')
parser.add_argument("--attn-dim", default=None,
                    type=int, action='store',
                    help='Number of neurons in to use in attention.'
                         'None means attn-dim = hidden-dim')
parser.add_argument("--embedding-dim", default=16,
                    type=int, action='store',
                    help='Embedding dimension')
parser.add_argument("--depth", default=1,
                    type=int, action='store',
                    help='Number of hidden layers in encoder and decoder')
parser.add_argument("--attn-type", default='bahdanau',
                    type=str, action='store',
                    help="Attention type, either 'bahdanau' or 'luong'")
parser.add_argument("--attn-input-feeding", default=True,
                    type=bool, action='store',
                    help='Whether attention is fed to decoder inputs')
parser.add_argument("--use-residual", default=False,
                    type=bool, action='store',
                    help='Whether to use residual connection')
parser.add_argument("--reverse-source", default=True,
                    type=bool, action='store',
                    help='Whether to reverse encode input sequences')
parser.add_argument("--keep-every-n-hours", default=1,
                    type=int, action='store',
                    help='Checkpoint keep interval')

# Model training params (optional)
parser.add_argument("--max-seq-len", default=100,
                    type=int, action='store',
                    help='Maximum sequence length')

parser.add_argument("--optimizer", default='adam',
                    type=str, action='store',
                    help="Optimizer to use, should be "
                         "'adadelta', 'adam' or 'rmsprop'")
parser.add_argument("--learning-rate", default=0.0001,
                    type=float, action='store',
                    help='Learning rate of the optimizer')
parser.add_argument("--max-gradient-norm", default=1.0,
                    type=float, action='store',
                    help='Maximum norm to clip gradient')

parser.add_argument("--dropout-rate", default=0.2,
                    type=float, action='store',
                    help='Hidden layer dropout')

# Training params (optional)
parser.add_argument("--batch-size", default=32,
                    type=int, action='store',
                    help='Batch size to feed into model')
parser.add_argument("--validate-every-n-batch", default=100,
                    type=int, action='store',
                    help='Save model checkpoint every n batch')
parser.add_argument("--validate-n-rows", default=None,
                    type=int, action='store',
                    help='Number of rows to read from validation file')
parser.add_argument("--file-batch-size", default=8192,
                    type=int, action='store',
                    help='Number of rows to read in-memory from each file')
parser.add_argument("--max-file-pool-size", default=50,
                    type=int, action='store',
                    help='Maximum number of files to cycle at a time')
parser.add_argument("--shuffle-files", default=True,
                    type=bool, action='store',
                    help='Shuffle files before reading')
parser.add_argument("--shuffle-file-batches", default=True,
                    type=bool, action='store',
                    help='Shuffle file batches before training')
parser.add_argument("--save-every-n-batch", default=1000,
                    type=int, action='store',
                    help='Save model checkpoint every n batch')


def train_model(args):
    """Train sequence to sequence model with training files."""
    
    # Parse model parameters
    model_params = dict([(arg,val) for arg,val in vars(args).items() 
                         if arg in get_Seq2Seq_model_param_names()])
    
    # Model
    model = Seq2Seq(model_dir=args.model_dir, 
                    dict_path=args.dict_path,
                    **model_params)
    
    # Train
    if args.train_data_path is not None:
        
        # Train files
        files = get_path_files(args.train_data_path)
        
        if args.shuffle_files:
            np.random.shuffle(files)
    
        # Batch generators
        # File batches
        file_gen = read_files_cycled(
            filenames=files,
            max_file_pool_size=args.max_file_pool_size,
            file_batch_size=args.file_batch_size, 
            file_batch_shuffle=False)
        
        # Train batches
        train_gen = rebatch(
            file_gen, 
            in_batch_size_limit=args.file_batch_size*args.max_file_pool_size,
            out_batch_size=args.batch_size, 
            shuffle=args.shuffle_file_batches,
            flatten=True)
        
        if args.validation_data_path is not None:
            valid_data = read_file(args.validation_data_path,
                                   nrows=args.validate_n_rows)
            valid_source_docs,valid_target_docs = zip(*valid_data)
        
        # Train
        start = time.clock()
        for batch_nb,batch in enumerate(train_gen):
            source_docs,target_docs = zip(*batch)
            loss,global_step = model.train(
                source_docs, target_docs,
                dropout_rate=args.dropout_rate,
                optimizer=args.optimizer,
                learning_rate=args.learning_rate,
                max_gradient_norm=args.max_gradient_norm,
                max_seq_len=args.max_seq_len,
                save_every_n_batch=args.save_every_n_batch)
            
            # Print progress
            end = time.clock()
            samples = global_step*args.batch_size
            print('[{}] Training step: {} - Samples: {} - Loss: {:<.3f} - Time {:<.3f}'\
                .format(str(datetime.now()), global_step, samples, loss, round(end-start,3)))
            start = end
            
            # Validation
            if args.validation_data_path is not None:
                if batch_nb % args.validate_every_n_batch == 0 and batch_nb > 0:
                    loss,global_step = model.eval(valid_source_docs, valid_target_docs)
                    end = time.clock()
                    print('[{}] Validation step: {} - Samples: {} - Loss: {:<.3f} - Time {:<.3f}'\
                    .format(str(datetime.now()), global_step, samples, loss, round(end-start,3)))
                    start = end
    
    else:
        print('Model created, but no training files were provided!')
    
    
def main():
    args = parser.parse_args()
    train_model(args)
    
    
if __name__=='__main__':
    main()
    
    
