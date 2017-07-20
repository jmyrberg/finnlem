# -*- coding: utf8 -*-
'''
Created on 19.7.2017

@author: Jesse
'''
import pandas as pd
import tensorflow as tf
from train_control import TrainController
from model_train import get_model_config2
from dictionary import Dictionary
from data_utils import BatchGenerator
from models import Seq2SeqModel

class Preprocessor():
    
    def process_doc(self,doc):
        doc = list(doc)
        return(doc)
        
    def process_docs(self,docs):
        for doc in docs:
            yield self.process_doc(doc)


name = 'Morphology'
fname = './data/morph/in/morph.csv'
dict_path = './data/morph.dict'
run = False
train = False

# Fit dictionary
if run:
    docs = pd.read_csv(fname,encoding='utf8')
    d = Dictionary(preprocessor=Preprocessor(), 
                vocab_size=100000, 
                min_freq=0.0, 
                max_freq=1.0, 
                prune_every_n=5000)
    d.fit_batch(docs.input.values)
    d.fit_batch(docs.target.values)
    d.lock()
    d.save(dict_path)

# TrainController
tc = TrainController(name, dict_path=dict_path)

if train:
    # Train
    if len(tc.to_train_files) < 1:
        print('Adding train files')
        tc.add_train_files([fname]*100, allow_copies=True)
    tc.train(batch_size=32, max_seq_len=1000, 
             file_batch_size=2000000, save_every_n_batch=50)
elif not train:
    # Predict
    #docs = pd.read_csv(fname,encoding='utf8',nrows=50)
    tc.predict([
        
        'koira','koiran','koiraa',
        'koirana','koiraksi','koirassa','koirasta','koiraan',
        
        'kissa','kissan','kissaa',
        'kissana','kissaksi','kissassa','kissasta','kissaan',
        
        'gepardi','gepardin','gepardia','gepardina','gepardiksi',
        'gepardistaansakohaan',
        
        'infrastruktuuritukeansakin'
        
        ], batch_size=1)
