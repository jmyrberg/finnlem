# -*- coding: utf8 -*-
'''
Created on 13.7.2017

@author: Jesse
'''
from keras.preprocessing.sequence import pad_sequences

import numpy as np
from utils.data_utils import read_files_cycled, rebatch

def process_doc(df,dict_vocab):
    """Clean words in a one or two-column dataframe."""
    if df.shape[1] == 1:
        print('SHAPE == 1')
        df['_NO_TARGET_'] = '<NO_TARGET>'
        print('Shape after',df.shape)
    print(df.shape)
    print(df.shape[1] == 1)
    df = df.dropna()
    for icol in [0,1]:
        print('Start shape',df.shape)
        # Convert accents apart from ä,ö
        valid_inds = ~df.iloc[:,icol].str.contains(r'ä|ö')
        df.loc[valid_inds,df.columns[icol]] = \
            df.loc[valid_inds,df.columns[icol]]\
                                    .str.normalize('NFKD')\
                                    .str.encode('ascii', errors='ignore')\
                                    .str.decode('utf-8')
        # Lowercase and strip
        df.iloc[:,icol] = df.iloc[:,icol].str.lower().str.strip()
        # Remove non-vocab characters
        df.iloc[:,icol] = df.iloc[:,icol] \
                .apply(lambda x: "".join([c for c in x if c in dict_vocab]))
        # Filter out rows with less than 3 letters
        df = df[df.iloc[:,icol].str.replace(r'\-|\#','').str.len() > 2]
        # Keep rows with length less than or equal to 50
        df = df[df.iloc[:,icol].str.len() <= 50]
        print('End shape',icol,df.shape)
    return df.values

def doc2seq(ar,dictionary):
    source_lens,source_seqs = zip(*list(
                        dictionary.docs2seqs(ar[:,0],return_length=True)))
    target_lens,target_seqs = zip(*list(
                        dictionary.docs2seqs(ar[:,1],return_length=True)))
    return source_seqs,source_lens,target_seqs,target_lens

def process_file_batch(df,dictionary,return_original=False):
    dict_vocab = list(dictionary.token2id.keys())
    doc_ar = process_doc(df,dict_vocab)
    source_seqs,source_lens,target_seqs,target_lens = doc2seq(doc_ar,dictionary)
    if not return_original:
        return list(zip(source_seqs,source_lens,target_seqs,target_lens))
    else:
        return list(zip(doc_ar,source_seqs,source_lens,target_seqs,target_lens))

def pad_seqs(seqs,lens,dictionary):
    maxlen = max(lens)
    seqs = pad_sequences(seqs,
                          maxlen=maxlen,
                          padding='post',
                          truncating='post',
                          value=dictionary.PAD)
    return seqs

def process_train_batch(source_seqs, source_lens,
                        target_seqs, target_lens, 
                        dictionary):
    source_lens = np.array(source_lens, dtype=np.int32).flatten()
    target_lens = np.array(target_lens, dtype=np.int32).flatten()
    source_seqs = pad_seqs(source_seqs,source_lens,dictionary)
    target_seqs = pad_seqs(target_seqs,target_lens,dictionary)
    return source_seqs,source_lens,target_seqs,target_lens

def generate_train_batches(filenames, dictionary,
                           batch_size=32,
                           max_file_pool_size=8,
                           file_batch_size=8192):
    file_gen = read_files_cycled(filenames,
                                 max_file_pool_size=max_file_pool_size,
                                 file_batch_size=file_batch_size,
                                 file_batch_shuffle=False,
                                 return_mode='df')
    for file_batch in file_gen:
        file_batch = process_file_batch(file_batch, dictionary,
                                        return_original=False)
        for train_batch in rebatch(file_batch,
                       in_batch_size_limit=max_file_pool_size*file_batch_size,
                       out_batch_size=batch_size,
                       shuffle=True):
            source_seqs,source_lens,target_seqs,target_lens = zip(*train_batch)
            batch = process_train_batch(source_seqs,source_lens,
                                        target_seqs,target_lens,
                                        dictionary)
            yield batch
            
def process_decode_batch(source_seqs,source_lens,dictionary):
    source_lens = np.array(source_lens, dtype=np.int32).flatten()
    source_seqs = pad_seqs(source_seqs,source_lens,dictionary)
    return source_seqs,source_lens
                
def generate_decode_batches(filenames, dictionary,
                           batch_size=32,
                           max_file_pool_size=8,
                           file_batch_size=8192,
                           shuffle=False):
    file_gen = read_files_cycled(filenames,
                                 max_file_pool_size=max_file_pool_size,
                                 file_batch_size=file_batch_size,
                                 file_batch_shuffle=False,
                                 return_mode='df')
    for file_batch in file_gen:
        file_batch = process_file_batch(file_batch, dictionary,
                                        return_original=True)
        for decode_batch in rebatch(file_batch,
                        in_batch_size_limit=max_file_pool_size*file_batch_size,
                        out_batch_size=batch_size,
                        shuffle=shuffle):
            (doc_ar,source_seqs,source_lens,
                target_seqs,target_lens) = zip(*decode_batch)
            source_seqs,source_lens = process_decode_batch(source_seqs,source_lens,
                                         dictionary)
            yield (doc_ar,source_seqs,source_lens,target_seqs,target_lens)
            