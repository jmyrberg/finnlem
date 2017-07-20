# -*- coding: utf8 -*-
'''
Created on 13.7.2017

@author: Jesse
'''
from model_processing import Preprocessor
from dictionary import Dictionary
from data_utils import read_files_batched
from utils import list_files_in_folder

# Config
cfg = {}
cfg['save_path'] = './data/dictionary.dict'

# Preprocessor
cfg['pp.tokenize'] = True
cfg['pp.remove_punctuation'] = True
cfg['pp.stem'] = False
cfg['pp.smart_lower'] = True
cfg['pp.ignore_stopwords'] = False
cfg['pp.minlen'] = 2

# Dictionary
cfg['d.vocab_size'] = 100000
cfg['d.min_freq'] = 0.1
cfg['d.max_freq'] = 1.0
cfg['d.prune_every_n'] = 5000

# Training
cfg['tr.files'] = list_files_in_folder('./data/feed/processed2')
cfg['tr.cols'] = ['target']
cfg['tr.file_batch_size'] = 1000
cfg['tr.file_batch_shuffle'] = False
cfg['tr.max_batches'] = None

def get_params(cfg,prefix):
    params = dict((k.split('.')[-1],v) for k,v in cfg.items() if prefix+'.' in k)
    return(params)

def train_dict(cfg):
    
    # Preprocessor
    pp_params = get_params(cfg,'pp')
    pp = Preprocessor(**pp_params)
    
    # Dictionary
    d_params = get_params(cfg,'d')
    d_params['preprocessor'] = pp
    d = Dictionary(**d_params)
    
    # Train
    tr_params = get_params(cfg,'tr')
    fs = read_files_batched(**tr_params)
    
    for _,docs in fs:
        d.fit_batch(docs, verbose=True)
    d.lock()
    
    # Save
    d.save(cfg['save_path'])
    d._print_debug()

    
train_dict(cfg)