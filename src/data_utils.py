# -*- coding: utf8 -*-
'''
Created on 11.7.2017

@author: Jesse
'''
import fileinput
import inspect
import math
import time
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import itertools
from collections import deque
from collections import defaultdict
from utils import list_files_in_folder
    
SEED = 2017
    
def batchify(it, batch_size=32, shuffle=False, max_batches=None):
    """Return iterable in batches."""
    batch_nb = 0
    batch = []
    for entry in it:
        batch.append(entry)
        batch_nb += 1
        if batch_nb % batch_size == 0:
            if shuffle:
                np.random.shuffle(batch)
            yield batch
            batch = []
            if max_batches is not None:
                if batch_nb >= max_batches:
                    break
    if len(batch) > 0:
        yield batch
    
def read_file_batched(filename, 
                      file_batch_size=8192, 
                      file_batch_shuffle=False, 
                      max_batches=math.inf,
                      return_mode='array', 
                      pd_kwargs={}):
    """Read file in batches."""
    batch_iterator = pd.read_csv(filename, 
                                 chunksize=file_batch_size,
                                 **pd_kwargs)
    for batch_nb,batch_df in enumerate(batch_iterator):
        if batch_nb+1 <= max_batches:
            
            if file_batch_shuffle:
                batch_df = batch_df.sample(frac=1, random_state=SEED)
            
            if return_mode == 'array':
                yield batch_df.values
            elif return_mode == 'df':
                yield batch_df
            elif return_mode == 'list':
                yield batch_df.values.tolist()
            elif return_mode == 'dict_records':
                yield batch_df.to_dict('records')
            elif return_mode == 'dict_list':
                yield batch_df.to_dict('list')
     
def read_files_batched(filenames,
                       file_batch_size=8192, 
                       file_batch_shuffle=False, 
                       max_batches=math.inf,
                       return_mode='array', 
                       n_jobs=-1,
                       max_batches_in_queue=1000,
                       max_queue_wait_seconds=0.5,
                       pd_kwargs={}):
    """Read multiple files in parallel."""
    def listify_generator(func,*args,**kwargs):
        listified_generator = list(func(*args,**kwargs))
        return(listified_generator)
    
    if n_jobs == -1:
        n_jobs = cpu_count()-1
        n_jobs = min((n_jobs,len(filenames)))

    # Parallel
    if n_jobs > 1:

        # Batch queue, appended in callback
        batch_queue = deque(maxlen=max_batches_in_queue)
        def callback(batch):
            while True:
                if len(batch_queue) < max_batches_in_queue:
                    batch_queue.append(batch)
                    break
                else:
                    time.sleep(0.1)
        
        # Create processes
        p = Pool(n_jobs)
        for filename in filenames:
            p.apply_async(listify_generator,
                          (read_file_batched,filename),
                          dict(file_batch_size=file_batch_size,
                               file_batch_shuffle=file_batch_shuffle, 
                               max_batches=max_batches,
                               return_mode=return_mode,
                                pd_kwargs=pd_kwargs),
                          callback=callback)
       
        # Yield from queue    
        keep_trying = True
        last_non_empty_batch = None
        while keep_trying:
            if len(batch_queue) > 0:
                for batch in batch_queue.popleft():
                    yield batch
                last_non_empty_batch = time.clock()
            
            if len(batch_queue) == 0:
                if last_non_empty_batch is not None:
                    if time.clock()-last_non_empty_batch >= max_queue_wait_seconds:
                        keep_trying = False    
        p.close()
        p.join()
    
    # Single process
    else:
        for filename in filenames:
            for batch in read_file_batched(filename,
                                           file_batch_size=file_batch_size,
                                           file_batch_shuffle=file_batch_shuffle, 
                                           max_batches=max_batches,
                                           return_mode=return_mode,
                                           pd_kwargs=pd_kwargs):
                yield batch
        
def read_files_cycled(filenames, 
                         max_file_pool_size=8,
                         file_batch_size=8192,
                         file_batch_shuffle=False,
                         max_batches=math.inf,
                         return_mode='array', 
                         pd_kwargs={}):
    """Cycle over multiple files at once."""
    file_pool = []
    files_done_pool = []
    n_files_in_pool = 0
    n_files_left = len(filenames)
    max_file_pool_size = min((max_file_pool_size,n_files_left))
    n_stop_iters = 0
    max_stop_iters = max_file_pool_size
    while n_files_left > 0:
        while n_files_in_pool < max_file_pool_size and n_files_left > 0:
            new_filename = filenames.pop()
            new_batch_gen = read_file_batched(new_filename, 
                                              file_batch_size=128)
            file_pool.append((new_filename,new_batch_gen))
            n_files_in_pool = len(file_pool)
            n_files_left = len(filenames)
            print('Added new file %s in pool (size %s)' % \
                  (new_filename,n_files_in_pool))
            print('Files left:',n_files_left)

        for i,(filename,batch_gen) in enumerate(itertools.cycle(file_pool)):
            try:
                batch = next(batch_gen)
                yield i%n_files_in_pool,len(batch)
            except StopIteration:
                remove_idx = [i for i,(f,_) in enumerate(file_pool)\
                              if f == filename][0]
                files_done_pool.append(file_pool.pop(remove_idx))
                n_files_in_pool -= 1
                print('File %s ended, pool size %d' % \
                      (filename,n_files_in_pool))
                if n_files_left > 0:
                    break
                else:
                    n_stop_iters += 1
                    if n_stop_iters < max_stop_iters:
                        continue
                    else:
                        break
        
    
for k in read_files_sequenced(list_files_in_folder('./data/feed/processed2/IS/')[:8]+\
                              list_files_in_folder('./data/feed/processed')):
    print(k)
    

class FuncGenerator():
    """Pass generator through a function and return results as a generator"""
    
    def __init__(self,func,gen):
        self.func = func
        self.gen = gen

    def __next__(self):
        for entry in self.gen:
            return self.func(entry)
        raise StopIteration

    def __iter__(self):
        return self
            
    def __repr__(self):
        return('<Generator object with func "%s" applied to iterable "%s">' % \
               (self.func,self.it))

class DataPipeline():
    """Pass iterable or generator through multiple functions."""
    
    layers = defaultdict(list)
    n_layers = 0
        
    def _apply_func(self,func,data):
        if inspect.isgenerator(data) or isinstance(data,FuncGenerator):
            return FuncGenerator(func,data)
        else:
            return func(data)
        
    def add(self,func,name=None):
        if name is None:
            name = str(self.n_layers+1).zfill(2)+'_'+func.__name__
        self.layers[self.n_layers].append((name,func))
        self.n_layers += 1
        
    def run(self,data):
        self.data_ = data
        for i in range(self.n_layers):
            layer = self.layers[i]
            for _,func in layer:
                self.data = self._apply_func(func,self.data_)
        return(self.data)
            
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
