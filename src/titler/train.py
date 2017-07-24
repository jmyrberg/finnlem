# -*- coding: utf8 -*-
'''
Created on 19.7.2017

@author: Jesse
'''
import os

import lemmer.decode

from dictionary.dictionary import load_dict
from titler.preprocessing import generate_train_batches
from models.seq2seq import Seq2Seq
from utils.utils import list_files_in_folder

# Lemmer
lemmer_dict_path = '../data/lemmer/dicts/dictionary.dict'
lemmer_model_dir = '../data/lemmer/models/test_lemmer/'

# Titler
titler_dict_path = '../data/titler/dicts/dictionary.dict'
titler_model_dir = '../data/models/test_titler/'

# Train
train_data_path = '../data/titler/train_data/'

optimizer = 'adam'
learning_rate = 0.0001
dropout_rate = 0.2

batch_size = 64
max_file_pool_size = 10
file_batch_size = 2500000

save_every_n = 200
keep_every_n_hours = 1


def train_titler():
    
    # Train a single file or multiple
    if os.path.isfile(train_data_path):
        files = [train_data_path]
    else:
        files = list_files_in_folder(train_data_path)
    
    # Lemmer
    lemmer_model,lemmer_dict = lemmer.decode.get_model_and_dict(
                                        model_dir=lemmer_model_dir, 
                                        dict_path=lemmer_dict_path)
    
    # Titler model
#     titler_model = Seq2Seq(mode='train',
#                            model_dir=titler_model_dir,
#                            dropout_rate=dropout_rate,
#                            optimizer=optimizer,
#                            learning_rate=learning_rate,
#                            keep_every_n_hours=keep_every_n_hours)
#     
#     # Titler dictionary
    titler_dict = load_dict(titler_dict_path)
    
    titler_batch_generator = generate_train_batches(files,
                                                    titler_dict,None,
                                                    lemmer_dict,lemmer_model)
    
    
train_titler()
    
    
