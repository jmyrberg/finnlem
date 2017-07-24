# -*- coding: utf8 -*-
'''
Created on 22.7.2017

@author: Jesse
'''
import os
import lemmer.decode
import lemmer.preprocessing

from titler.preprocessing import get_tokens_from_lemmer_seqs
from dictionary.dictionary import Dictionary
from utils.utils import list_files_in_folder

# Lemmer
lemmer_dict_path = '../data/lemmer/dicts/dictionary.dict'
lemmer_model_dir = '../data/lemmer/models/test_lemmer/'

# Titler
vocab_size = 100000
min_freq = 0.0
max_freq = 1.0
titler_dict_vocab_train_path = '../data/titler/dict_train/'
titler_dict_vocab_path = 'D:/Koodaus/EclipseWS/ClickSaver/src/data/titler/dicts/dictionary.vocab',
titler_dict_path = '../data/titler/dicts/dictionary.dict'

# Training
batch_size = 2048
file_batch_size = 8192*2

def train_titler_vocab_and_dict():
   
    # Create a dictionary
    titler_dict = Dictionary(vocab_size=vocab_size,
                             min_freq=min_freq,
                             max_freq=max_freq)
    
    # Words --> Chars (lemmer) --> Lemmatized words --> Sentences
    train_files = []
    if os.path.isfile(titler_dict_vocab_train_path):
        train_files = [titler_dict_vocab_train_path]
    else:
        train_files = list_files_in_folder(titler_dict_vocab_train_path)

    # Convert words to their base form
    lemmer_model,lemmer_dict = lemmer.decode.get_model_and_dict(
                                        model_dir=lemmer_model_dir, 
                                        dict_path=lemmer_dict_path)

    lemmer_decode_generator = lemmer.preprocessing.generate_decode_batches(
                                        filenames=train_files, 
                                        dictionary=lemmer_dict,
                                        batch_size=batch_size,
                                        file_batch_size=file_batch_size*2, 
                                        shuffle=False)

    # Lemmatize
    for (doc_ar, source_seqs, source_lens,
            target_seqs, target_lens) in lemmer_decode_generator:
        pred_words = get_tokens_from_lemmer_seqs(source_seqs,source_lens,
                                                 lemmer_dict,lemmer_model)
        
        titler_dict.fit_batch(pred_words,
                              prune_every_n=1000)
        
        #=======================================================================
        # for real,pred in zip(doc_ar,pred_words):
        #     print(real[0],'-->',"".join(pred))
        #     
        # 
        # if titler_dict.n_tokens > 100000:
        #     titler_dict.lock()
        #     titler_dict.save(dict_path)
        #     break
        #=======================================================================

    # Save dict
    titler_dict.save(titler_dict_path)
        
    # Save vocab
    words = sorted(list(titler_dict.token2id))
    with open(titler_dict_vocab_path,'w',encoding='utf8') as f:
        for word in words:
            f.write(word+'\n')

#train_titler_dict()