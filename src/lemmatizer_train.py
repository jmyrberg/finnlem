# -*- coding: utf8 -*-
'''
Created on 19.7.2017

@author: Jesse
'''
import os
from dictionary import Dictionary
from train_utils import TrainController
from lemmatizer_preprocessing import generate_train_batches, \
                                generate_decode_batches

mode = 'decode'

# Dictionary
dict_train = True
dict_vocab_path = './data/lemmatizer/dicts/dictionary.vocab'
dict_path = './data/lemmatizer/dicts/dictionary.dict'

# Lemmatizer
lemmatizer_name = 'TestLemmatizer2'
lemmatizer_dir = './data/lemmatizer/controllers/'

# Model config
mc = {}

mc['cell_type'] = 'lstm'
mc['hidden_dim'] = 256
mc['embedding_dim'] = 128
mc['depth'] = 2

mc['attn_type'] = 'bahdanau'
mc['attn_input_feeding'] = True
mc['use_residual'] = True

mc['dropout_rate'] = 0.3

mc['beam_width'] = 2

# Train
train_batch_size = 64
train_file_batch_size = 2500000
opt_params = {}
opt_params['learning_rate'] = 0.0001

# Decode
decode_batch_size = 64

def dictionary():
    with open(dict_vocab_path,'r',encoding='utf8') as f:
        dict_vocab = f.read().splitlines()
        print(dict_vocab)
    if not os.path.exists(dict_path) or dict_train:
        d = Dictionary()
        d.fit([dict_vocab])
        d.save(dict_path)
    elif os.path.exists(dict_path):
        pass
    else:
        raise FileNotFoundError("Dictionary doesn't exist!")
            
def train():
    tc = TrainController(controller_name=lemmatizer_name,
                         base_dir=lemmatizer_dir,
                         dict_path=dict_path,
                         model_config=mc)
    if len(tc.to_train_files) < 1:
        tc.add_train_files(['D:/Koodaus/EclipseWS/ClickSaver/src/data/lemmatizer/train_data/all_lemmatizer.csv']*10, allow_copies=True)
    train_generator = generate_train_batches(filenames=tc.to_train_files,
                                             file_batch_size=train_file_batch_size,
                                             batch_size=train_batch_size,
                                             dictionary=tc.dictionary)
    tc.train(train_generator,opt_params=opt_params)
    
def decode():
    tc = TrainController(controller_name=lemmatizer_name,
                         base_dir=lemmatizer_dir,
                         dict_path=dict_path,
                         model_config=mc)
    decode_generator = generate_decode_batches(filenames=['./data/morph/in/examples.csv'],
                                               batch_size=decode_batch_size,
                                               dictionary=tc.dictionary)
    tc.decode(decode_generator)


def main():
    if mode == 'dictionary':
        dictionary()
    elif mode == 'train':
        train()
    elif mode == 'decode':
        decode()
    
    
if __name__ == '__main__':
    main()