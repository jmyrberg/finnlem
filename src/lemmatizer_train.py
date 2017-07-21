# -*- coding: utf8 -*-
'''
Created on 19.7.2017

@author: Jesse
'''
import os
import string
from dictionary import Dictionary
from train_utils import TrainController
from models import Seq2Seq
from utils import list_files_in_folder

# Dictionary
dict_train = False
dict_vocab = [list(string.ascii_lowercase) + ['ä','ö']]
dict_path = './data/lemmatizer/dicts/dictionary.dict'

# Lemmatizer
lemmatizer_name = 'TestLemmatizer'
lemmatizer_dir = './data/lemmatizer/training/'
lemmatizer_train = True

# Model config
mc = {}

mc['cell_type'] = 'lstm'
mc['hidden_dim'] = 256
mc['embedding_dim'] = 128
mc['depth'] = 2

mc['attn_type'] = 'bahdanau'
mc['attn_input_feeding'] = False
mc['use_residual'] = True

mc['dropout_rate'] = 0.0

mc['beam_width'] = 2


def train_dict():
    if not os.path.exists(dict_path) or dict_train:
        d = Dictionary()
        d.fit(dict_vocab)
        d.save(dict_path)
    elif os.path.exists(dict_path):
        pass
    else:
        raise FileNotFoundError("Dictionary doesn't exist!")

def train_model():
    tc = TrainController(controller_name=lemmatizer_name,
                         base_dir=lemmatizer_dir,
                         dict_path=dict_path,
                         model_config=mc)
    tc.add_train_files(list_files_in_folder('./data/feed/processed'))
    tc.train(opt_params={'learning_rate':0.002})

def train_lemmatizer():
    train_dict()
    train_model()

def main():
    train_lemmatizer()
    
if __name__ == '__main__':
    main()



# class Preprocessor():
#     
#     def process_doc(self,doc):
#         doc = list(doc)
#         return(doc)
#         
#     def process_docs(self,docs):
#         for doc in docs:
#             yield self.process_doc(doc)
# 
# 
# name = 'Morphology'
# fname = './data/morph/in/morph.csv'
# dict_path = './data/morph.dict'
# run = False
# train = False
# 
# # Fit dictionary2
# if run:
#     docs = pd.read_csv(fname,encoding='utf8')
#     d = Dictionary(preprocessor=Preprocessor(), 
#                 vocab_size=100000, 
#                 min_freq=0.0, 
#                 max_freq=1.0, 
#                 prune_every_n=5000)
#     d.fit_batch(docs.input.values)
#     d.fit_batch(docs.target.values)
#     d.lock()
#     d.save(dict_path)
# 
# # TrainController
# tc = TrainController(name, dict_path=dict_path)
# 
# if train:
#     # Train
#     if len(tc.to_train_files) < 1:
#         print('Adding train files')
#         tc.add_train_files([fname]*100, allow_copies=True)
#     tc.train(batch_size=32, max_seq_len=1000, 
#              file_batch_size=2000000, save_every_n_batch=50)
# elif not train:
#     # Predict
#     #docs = pd.read_csv(fname,encoding='utf8',nrows=50)
#     tc.predict([
#         
#         'koira','koiran','koiraa',
#         'koirana','koiraksi','koirassa','koirasta','koiraan',
#         
#         'kissa','kissan','kissaa',
#         'kissana','kissaksi','kissassa','kissasta','kissaan',
#         
#         'gepardi','gepardin','gepardia','gepardina','gepardiksi',
#         'gepardistaansakohaan',
#         
#         'infrastruktuuritukeansakin'
#         
#         ], batch_size=1)
