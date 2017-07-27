# -*- coding: utf8 -*-
'''
Created on 11.7.2017

@author: Jesse
'''
from collections import deque
import itertools
import math
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import time

import numpy as np
import pandas as pd

SEED = 2018
    
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
    
def rebatch(batches, 
            in_batch_size_limit=8192, 
            out_batch_size=32, 
            shuffle=True, 
            flatten=False):
    """Convert input batches to output batches with different size."""
    # Input
    in_batches = []
    in_batch_nb = 0
    out_batch = []
    for in_batch in batches:
        # Input
        if flatten:
            in_batches.extend(in_batch)
            in_batch_nb += len(in_batch)
        else:
            in_batches.append(in_batch)
            in_batch_nb += 1
            
        # Shuffle just once per input batch
        if in_batch_nb >= out_batch_size:
            if shuffle:
                np.random.shuffle(in_batches)
                
        # Output
        while in_batch_nb >= out_batch_size:
            for out_batch in batchify(in_batches,
                                      batch_size=out_batch_size,
                                      shuffle=False):
                yield out_batch
                
                in_batches = in_batches[in_batch_size_limit:]
                in_batch_nb = len(in_batches)
    
    if len(in_batches) > 0:
        if shuffle:
            np.random.shuffle(in_batches)
        yield out_batch
    
def read_file(filename, nrows=None):
    """Read one file entirely."""
    ar = pd.read_csv(filename, nrows=nrows).values
    return ar
    
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
                      max_batches_per_file=math.inf,
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
    force_loop_restart = True
    while n_files_left > 0 or force_loop_restart:
        
        force_loop_restart = False
        
        # Add new file(s)
        while n_files_in_pool < max_file_pool_size and n_files_left > 0:
            new_filename = filenames.pop()
            new_batch_gen = read_file_batched(new_filename, 
                                              file_batch_size=file_batch_size,
                                              file_batch_shuffle=file_batch_shuffle,
                                              max_batches=max_batches_per_file,
                                              return_mode=return_mode,
                                              pd_kwargs=pd_kwargs)
            file_pool.append((new_filename,new_batch_gen))
            n_files_in_pool = len(file_pool)
            n_files_left = len(filenames)
            
        # Cycle until new files needs to be added or we reach the end of pool
        for filename,batch_gen in itertools.cycle(file_pool):
            try:
                batch = next(batch_gen)
                yield batch
            except StopIteration:
                remove_idx = file_pool.index((filename,batch_gen))
                files_done_pool.append(file_pool.pop(remove_idx))
                n_files_in_pool = len(file_pool)
                if n_files_left > 0:
                    break
                else:
                    n_stop_iters += 1
                    if n_stop_iters < max_stop_iters:
                        force_loop_restart = True
                        break
                    else:
                        break
