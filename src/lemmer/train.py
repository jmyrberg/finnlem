# -*- coding: utf8 -*-
'''
Created on 19.7.2017

@author: Jesse
'''
import os

from dictionary.dictionary import load_dict
from preprocessing import generate_train_batches
from models.seq2seq import Seq2Seq
from utils.utils import list_files_in_folder

# Dictionary
dict_path = '../data/lemmer/dicts/dictionary.dict'

# Model
mode = 'train'
model_dir = '../data/trainFolder/'

# Train
train_data_path = '../data/lemmer/train_data/'

optimizer = 'adam'
learning_rate = 0.0001
dropout_rate = 0.2

batch_size = 64
max_file_pool_size = 10
file_batch_size = 2500000

save_every_n = 200
keep_every_n_hours = 1


def train_lemmer():
    
    # Init
    model = Seq2Seq(mode=mode,
                    model_dir=model_dir,
                    dropout_rate=dropout_rate,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    keep_every_n_hours=keep_every_n_hours)
    dictionary = load_dict(dict_path)
    
    # Train a single file or multiple
    if os.path.isfile(train_data_path):
        files = [train_data_path]
    else:
        files = list_files_in_folder(train_data_path)
    
    # Get generator
    train_generator = generate_train_batches(filenames=files,
                                             dictionary=dictionary, 
                                             batch_size=batch_size, 
                                             max_file_pool_size=max_file_pool_size, 
                                             file_batch_size=file_batch_size)
    for (source_seqs, source_lens,
         target_seqs, target_lens) in train_generator:
        loss, global_step = model.train(source_seqs, source_lens,target_seqs, target_lens)
        
        print('Global step %d (%d samples) loss: %f' % \
              (global_step,global_step*batch_size,loss))
        
        if global_step % save_every_n == 0:
            model.save()
    
train_lemmer()
