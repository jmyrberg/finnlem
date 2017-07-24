# -*- coding: utf8 -*-
'''
Created on 13.7.2017

@author: Jesse
'''
import numpy as np
import pandas as pd

import lemmer.decode
import lemmer.preprocessing

from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize

from utils.data_utils import read_files_cycled, rebatch

def get_tokens_from_lemmer_seqs(seqs,seq_lens,lemmer_dict,lemmer_model):
    seqs = lemmer_model.decode(seqs,seq_lens)
    seqs = seqs[:,:,0] # Take "best" if BeamSearch
    
    words = lemmer_dict.seqs2docs(seqs)
    words = [["".join([c for c in word_chars if c != '#'])] 
                  for word_chars in words]
    return(words)

def tokenize_series_sentences(s):
    """Convert sentences in a series into words."""
    words = s.apply(lambda x: word_tokenize(x))
    word_lens = words.apply(lambda x: len(x)).values
    total_len = word_lens.sum()
    idx_ar = np.zeros(total_len,dtype=np.int32)
    start = 0
    for i,ellen in enumerate(idx_ar):
        end = start + ellen
        idx_ar[start:end] = i
        start = end
    words = words.sum() # Concatenate lists
    words_df = pd.DataFrame(words, index=idx_ar, columns=[s.name])
    return(words_df)

def process_file_batch(df, titler_dict, titler_model,
                       lemmer_dict, lemmer_model):
    df = df.dropna()
    
    # Convert sentences to words
    source_df = tokenize_series_sentences(df.iloc[:,0])
    target_df = tokenize_series_sentences(df.iloc[:,1])
    
    # Feed sentences into lemmer
    source_batches = lemmer.preprocessing.process_file_batch(
                                                    source_df, lemmer_dict)
    (source_seqs,source_lens,_,_) = zip(*source_batches)
    target_batches = lemmer.preprocessing.process_file_batch(
                                                    target_df, lemmer_dict)
    (target_seqs,target_lens,_,_) = zip(*target_batches)
    print('Lemmatizing sources...')
    (source_seqs,source_lens,
     target_seqs,target_lens) = lemmer.preprocessing.process_train_batch(source_seqs,source_lens,
                                                     target_seqs,target_lens,
                                                     lemmer_dict)
    print(source_seqs)
    source_pred_words = get_tokens_from_lemmer_seqs(source_seqs,source_lens,
                                                    lemmer_dict,lemmer_model)
    target_pred_words = get_tokens_from_lemmer_seqs(target_seqs,target_lens,
                                                    lemmer_dict,lemmer_model)
    print(source_pred_words)
    
    
def generate_train_batches(filenames, titler_dict, titler_model,
                           lemmer_dict, lemmer_model):
    file_gen = read_files_cycled(filenames,
                                 file_batch_size=4,
                                 file_batch_shuffle=False,
                                 return_mode='df')
    for file_batch in file_gen:
        file_batch = process_file_batch(file_batch,
                                        titler_dict, titler_model,
                                        lemmer_dict, lemmer_model)
        #for decode_batch in rebatch(file_batch,
        #                in_batch_size_limit=max_file_pool_size*file_batch_size,
        #                out_batch_size=batch_size,
        #                shuffle=shuffle):
        #    (doc_ar,source_seqs,source_lens,
        #        target_seqs,target_lens) = zip(*decode_batch)
        #    source_seqs,source_lens = process_decode_batch(source_seqs,source_lens,
        #                                 dictionary)
        #   yield (doc_ar,source_seqs,source_lens,target_seqs,target_lens)