# -*- coding: utf8 -*-
"""Sequence to sequence model training."""


import time
import argparse

from datetime import datetime

import numpy as np

from seq2seq import Seq2Seq
from utils.utils import get_path_files
from utils.data_utils import read_files_cycled, rebatch


parser = argparse.ArgumentParser('Model training')

# Path params (required)
parser.add_argument("--model-dir", required=True,
                    type=str, action='store',
                    help='Model checkpoint and log save path')

# Path params (required for the first time)
parser.add_argument("--dict-path", default=None,
                    type=str, action='store',
                    help='Path to existing Dictionary')

# Path params (optional)
parser.add_argument("--train-data-path", default=None,
                    type=str, action='store',
                    help='Training data folder or file')

# Model params (required for the first time)
# keep_every_n_hours = 1

# Model training params (optional)
parser.add_argument("--optimizer", default='adam',
                    type=str, action='store',
                    help='Optimizer to use')
parser.add_argument("--learning-rate", default=0.0001,
                    type=int, action='store',
                    help='Learning rate of the optimizer')
parser.add_argument("--dropout-rate", default=0.2,
                    type=str, action='store',
                    help='Hidden layer dropout')

# Training params (optional)
parser.add_argument("--batch-size", default=32,
                    type=int, action='store',
                    help='Batch size to feed into model')
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
parser.add_argument("--save-every-n", default=500,
                    type=int, action='store',
                    help='Maximum number of files to cycle at a time')


def train_model(args):
    """Train sequence to sequence model with training files."""
    
    # Model
    model = Seq2Seq(model_dir=args.model_dir, dict_path=args.dict_path)
    
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
        
        # Train
        start = time.clock()
        for batch_nb,batch in enumerate(train_gen):
            source_docs,target_docs = zip(*batch)
            loss,global_step = model.train(source_docs, target_docs,
                                           max_seq_len=args.max_seq_len)
            
            end = time.clock()
            print('[{}] Step: {} - Samples: {} - Loss: {:<.3f} - Time {:<.3f}'.format(
            str(datetime.now()), global_step, global_step*args.batch_size,
            loss,round(end-start,3)))
            start = end
        
            if batch_nb % 512 == 0 and batch_nb > 0:
                print('Evaluating...')
                print('Source:',source_docs[0])
                print('Target:',target_docs[0])
                print('Prediction:',model.decode(source_docs[0:1]))
    
    else:
        print('No training files provided')
    
    
    
def main():
    args = parser.parse_args()
    train_model(args)
    
if __name__=='__main__':
    main()
    
    
