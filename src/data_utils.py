# -*- coding: utf8 -*-
'''
Created on 11.7.2017

@author: Jesse
'''
import csv
import math
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from utils import list_files_in_folder
from _collections import defaultdict
    
def read_file_batched(filename, cols=None, file_batch_size=4096,
                      file_batch_shuffle=False, max_batches=None):
    batch_data = []
    batch_nb = 0
    batch_sample_nb = 0
    with open(filename,'r',encoding='utf8') as f:
        reader = csv.DictReader(f)
        if cols is None:
            cols = reader.fieldnames
        n_cols = len(cols)
        
        for r in reader:
            if n_cols > 1:
                batch_data.append(tuple(r[col] for col in cols))
            elif n_cols == 1:
                batch_data.append(r[cols[0]])
            batch_sample_nb += 1
            
            if batch_sample_nb % file_batch_size == 0:
                if file_batch_shuffle:
                    np.random.shuffle(batch_data)
                yield batch_data
                
                batch_data = []
                batch_nb += 1
                batch_sample_nb = 0
                
                if max_batches is not None:
                    if batch_nb >= 1:
                        break
        else:
            if len(batch_data) > 0:
                if file_batch_shuffle:
                    np.random.shuffle(batch_data)
                yield batch_data
     
def read_files_batched(files, cols=None, file_batch_size=4096,
                       file_batch_shuffle=False, max_batches=None):
    for filename in files:
        file_stream = read_file_batched(filename, cols, file_batch_size, 
                                        file_batch_shuffle, max_batches)
        for filename,batch_data in file_stream:
            yield filename,batch_data
            
            
class BatchGenerator():
    
    def __init__(self, dictionary):
        self.dictionary = dictionary
            
    def _pad_batch(self,batch,maxlen):
        batch = pad_sequences(batch,
                              maxlen=maxlen,
                              padding='post',
                              truncating='post',
                              value=self.dictionary.PAD)
        return(batch)
            
    def _process_train_batch(self, input_batch, input_batch_maxlen, input_batch_lens,
                       target_batch, target_batch_maxlen, target_batch_lens, 
                       max_seq_len):
        input_maxlen = min((input_batch_maxlen,max_seq_len))
        input_batch = self._pad_batch(input_batch,input_maxlen)
        input_batch_lens = np.array(input_batch_lens)
        target_maxlen = min((target_batch_maxlen,max_seq_len))
        target_batch = self._pad_batch(target_batch,target_maxlen)
        target_batch_lens = np.array(target_batch_lens)
        return (input_batch, input_batch_lens, target_batch, target_batch_lens)
                    
    def train_from_files(self, files, batch_size=32, max_seq_len=10000,
                    file_batch_size=4096, file_batch_shuffle=True,
                    shuffle_files=True, verbose=False, output_list=None):
        if shuffle_files:
            np.random.shuffle(files)
        for file_nb,filename in enumerate(files): # Iterate files
            file_batch_stream = read_file_batched(filename=filename, 
                                                  cols=['input','target'],
                                                  file_batch_size=file_batch_size,
                                                  file_batch_shuffle=file_batch_shuffle)
            file_batch_nb = 0
            file_sample_nb = 0
            batch_sample_nb = 0
            batch_iterated_sample_nb = 0
            batch_nb = 0
            input_batch = []
            input_batch_lens = []
            target_batch = []
            target_batch_lens = []
            input_batch_maxlen = -1
            target_batch_maxlen = -1
            # Iterate batches from file, size <file_batch_size>
            for file_batch in file_batch_stream: 
                        
                for input_doc,target_doc in file_batch:
                
                    if verbose:
                        print('Filename: {} - File batch nb: {} - File sample nb: {} - Batch nb: {} - Batch sample nb: {} - Batch iterated sample nb: {}'.format(
                            filename,file_batch_nb,file_sample_nb,batch_nb,batch_sample_nb,batch_iterated_sample_nb))
                
                    input_seq = self.dictionary.doc2seq(input_doc)#,
                                                        #prepend_SOS=True)
                    target_seq = self.dictionary.doc2seq(target_doc)#, 
                                                         #prepend_SOS=True, ei ollut
                                                         #append_EOS=True)
                    input_len = len(input_seq)
                    target_len = len(target_seq)
                    
                    # Validate train data
                    if input_len > 0 and target_len > 0:
                        input_batch.append(input_seq)
                        input_batch_lens.append(input_len)
                        input_batch_maxlen = max((input_batch_maxlen,input_len))
                        
                        target_batch.append(target_seq)
                        target_batch_lens.append(target_len)
                        target_batch_maxlen = max((target_batch_maxlen,target_len))
                        
                        batch_sample_nb += 1
                        
                    batch_iterated_sample_nb += 1
                    file_sample_nb += 1
                    
                    # Return batches of size <batch_size>
                    if batch_sample_nb % batch_size == 0 and batch_sample_nb > 0:
                        yield self._process_train_batch(
                            input_batch, input_batch_maxlen, input_batch_lens, 
                            target_batch, target_batch_maxlen, target_batch_lens, max_seq_len)
                    
                        input_batch = []
                        input_batch_lens = []
                        target_batch = []
                        target_batch_lens = []
                        input_batch_maxlen = -1
                        target_batch_maxlen = -1
                        batch_sample_nb = 0
                        batch_iterated_sample_nb = 0
                        batch_nb += 1
                        
                file_batch_nb += 1
                        
            # Last batch of a file
            if len(input_batch) > 0:
                yield self._process_train_batch(
                        input_batch, input_batch_maxlen, input_batch_lens, 
                        target_batch, target_batch_maxlen, target_batch_lens, max_seq_len)
                
                input_batch = []
                input_batch_lens = []
                target_batch = []
                target_batch_lens = []
                input_batch_maxlen = -1
                target_batch_maxlen = -1
                batch_sample_nb = 0
                batch_iterated_sample_nb = 0
                batch_nb += 1
                
            if output_list is not None:
                output_list.append(filename)
                
    def _process_predict_batch(self,batch,batch_lens,maxlen):
        batch = self._pad_batch(batch, maxlen=maxlen)
        batch_lens = np.array(batch_lens)
        return(batch,batch_lens)
                
    def predict_from_docs(self,docs,batch_size=32):
        input_batch = []
        input_batch_lens = []
        batch_sample_nb = 0
        maxlen = -1
        for batch_nb,doc in enumerate(docs):
            input_seq = self.dictionary.doc2seq(doc)#,prepend_SOS=True)
            input_len = len(input_seq)
            
            if input_len > 0:
                input_batch.append(input_seq)
                input_batch_lens.append(input_len)
                maxlen = max((input_len,maxlen))
                batch_sample_nb += 1
            
            if batch_sample_nb % batch_size == 0 and batch_sample_nb > 0:
                print("\n")
                print('Original:',doc)
                yield self._process_predict_batch(input_batch,input_batch_lens,maxlen)
                
                input_batch = []
                input_batch_lens = []
                batch_sample_nb = 0
                maxlen = -1