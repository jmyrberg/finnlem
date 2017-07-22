# -*- coding: utf8 -*-
'''
Created on 22.7.2017

@author: Jesse
'''
from dictionary.dictionary import Dictionary

dict_vocab_path = '../data/lemmer/dicts/dictionary.vocab',
dict_path = '../data/lemmer/dicts/dictionary.dict'

def train_lemmer_dict():
    # Read vocabulary
    with open(dict_vocab_path,'r',encoding='utf8') as f:
        dict_vocab = f.read().splitlines()
    
    # Fit dictionary
    d = Dictionary()
    d.fit([dict_vocab])
    d.save(dict_path)
    
train_lemmer_dict()