# -*- coding: utf8 -*-
'''
Created on 22.7.2017

@author: Jesse
'''
import os

import numpy as np

from titler import doc_to_tokens
from dictionary.dictionary import Dictionary
from utils.utils import list_files_in_folder
from utils.data_utils import read_files_batched

# Paths
titler_dict_path = '../data/titler/dicts/dictionary.dict'
titler_dict_vocab_train_path = '../data/titler/dict_train/'

# Dictionary
vocab_size = 100000
min_freq = 0.0
max_freq = 1.0

# Training
pd_kwargs = {'encoding':'iso-8859-1'}
file_batch_size = 8192*2
prune_every_n = 10

def train_dict():

    # Create a dictionary
    titler_dict = Dictionary(vocab_size=vocab_size,
                             min_freq=min_freq,
                             max_freq=max_freq)
    
    # Files to train
    train_files = []
    if os.path.isfile(titler_dict_vocab_train_path):
        train_files = [titler_dict_vocab_train_path]
    else:
        train_files = list_files_in_folder(titler_dict_vocab_train_path)

    # Batch generator
    train_gen = read_files_batched(train_files, 
                                   file_batch_size=file_batch_size,
                                   file_batch_shuffle=False,
                                   pd_kwargs=pd_kwargs)

    for docs in train_gen:
        long_doc = " ".join(docs.flatten())
        tokens = doc_to_tokens(long_doc)
        tokens = [[token] for token in tokens]
        titler_dict.fit_batch(tokens,
                              prune_every_n=prune_every_n)
    
    # Save dict
    titler_dict.lock()
    titler_dict.save(titler_dict_path)

#train_dict()