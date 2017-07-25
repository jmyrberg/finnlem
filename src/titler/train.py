# -*- coding: utf8 -*-
'''
Created on 19.7.2017

@author: Jesse
'''
import os
import time

import numpy as np

from datetime import datetime

from titler import Titler
from utils.utils import list_files_in_folder
from utils.data_utils import read_files_cycled, rebatch

# Titler
dict_path = '../data/titler/dicts/dictionary.dict'
model_dir = '../data/titler/models/test_titler2'

# Training
train_data_path = '../data/titler/train_data/'

optimizer = 'adam'
learning_rate = 0.0001
dropout_rate = 0.2

batch_size = 16
max_file_pool_size = 16
file_batch_size = 8192

save_every_n = 200
keep_every_n_hours = 1
max_seq_len = 3000


def train_titler():
    
    # Titler
    titler = Titler(model_dir=model_dir, dict_path=dict_path)
    
    # Train files
    if os.path.isfile(train_data_path):
        train_files = [train_data_path]
    else:
        train_files = list_files_in_folder(train_data_path)
    
    np.random.shuffle(train_files)
    
    for epoch_nb in range(100):
        print('Epoch number %d' % epoch_nb)
    
        # Batch generators
        # File batches
        file_gen = read_files_cycled(filenames=train_files,
                                      max_file_pool_size=max_file_pool_size, 
                                      file_batch_size=file_batch_size, 
                                      file_batch_shuffle=False)
        # Train batches
        train_gen = rebatch(file_gen, 
                            in_batch_size_limit=file_batch_size*max_file_pool_size,
                            out_batch_size=batch_size, 
                            shuffle=True,
                            flatten=True)
        
        # Train
        start = time.clock()
        for batch_nb,batch in enumerate(train_gen):
            source_docs,target_docs = zip(*batch)
            loss,global_step = titler.train(source_docs,target_docs,max_seq_len)
            
            end = time.clock()
            print('[{}] Step: {} - Samples: {} - Loss: {:<.3f} - Time {:<.3f}'.format(
            str(datetime.now()), global_step, global_step*batch_size,
            loss,round(end-start,3)))
            start = end
        
            if batch_nb % 512 == 0:
                print('Evaluating...')
                print('Source:',source_docs[0])
                print('Target:',target_docs[0])
                print('Prediction:',titler.decode(source_docs[0:1]))
    
train_titler()
    
    
