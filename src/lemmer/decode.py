# -*- coding: utf8 -*-
'''
Created on 22.7.2017

@author: Jesse
'''
import os

from dictionary.dictionary import load_dict
from preprocessing import generate_decode_batches
from models.seq2seq import Seq2Seq
from utils.utils import list_files_in_folder

# Dictionary
dict_path = '../data/lemmer/dicts/dictionary.dict'

# Model
mode = 'decode'
model_dir = '../data/trainFolder/'

# Decode
source_data_path = '../data/lemmer/test_data/'
output_path = '../data/lemmer/output/'
max_decode_step = 30
batch_size = 64
max_file_pool_size = 10
file_batch_size = 50000

def decode_lemmer():
    
    # Init
    model = Seq2Seq(mode=mode,
                    model_dir=model_dir,
                    max_decode_step=max_decode_step)
    dictionary = load_dict(dict_path)
    
    # Decode a single file or multiple
    if os.path.isfile(source_data_path):
        files = [source_data_path]
    else:
        files = list_files_in_folder(source_data_path)
    
    # Decode batch generator
    decode_generator = generate_decode_batches(filenames=files,
                                               dictionary=dictionary, 
                                               batch_size=batch_size, 
                                               max_file_pool_size=max_file_pool_size, 
                                               file_batch_size=file_batch_size)
    for (doc_ar, source_seqs, source_lens,
            target_seqs, target_lens) in decode_generator:
        pred_seqs = model.decode(source_seqs,source_lens)
        
        for (source_doc,target_doc),pred_seq in zip(doc_ar,pred_seqs):
            
            print('')
            print('{:>8s}: {:<50s}'.format('Source',source_doc))
            #print('{:>8s}: {:<50s}'.format('Target',target_doc))
            
            for k in range(pred_seq.shape[1]):
                pred_doc = dictionary.seq2doc(pred_seq[:,k])
                print('{:>5s} #{:>1s}: {:<50s}'.format('Pred',str(k+1),pred_doc))
            
decode_lemmer()