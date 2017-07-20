# -*- coding: utf8 -*-
'''
Created on 11.7.2017

@author: Jesse
'''
import math
import tensorflow as tf
from data_utils import BatchGenerator
from dictionary import Dictionary
from models import Seq2SeqModel
from utils import list_files_in_folder

def get_model_config(d):
    
    # Model config
    mc = {}
    mc['mode'] = 'train'
    mc['use_fp16'] = False
    
    # Network
    mc['cell_type'] = 'lstm'
    mc['hidden_units'] = 128
    mc['depth'] = 4
    mc['attention_type'] = 'luong'
    mc['embedding_size'] = 64
    mc['num_encoder_symbols'] = d.n_tokens
    mc['num_decoder_symbols'] = d.n_tokens
    mc['start_token'] = d.EOS
    mc['end_token'] = d.EOS
    mc['pad_token'] = d.PAD
    mc['use_residual'] = True
    mc['attn_input_feeding'] = True
    mc['use_dropout'] = True
    mc['dropout_rate'] = 0.0
    
    # Training
    mc['optimizer'] = 'adam'
    mc['learning_rate'] = 0.000001
    mc['max_gradient_norm'] = 1.0
    
    # Decoding
    mc['beam_width'] = 1
    mc['max_decode_step'] = 500
    return(mc)
    
def get_model_config2(d):
    
    # Model config
    mc = {}
    mc['mode'] = 'train'
    mc['use_fp16'] = False
    
    # Network
    mc['cell_type'] = 'lstm'
    mc['hidden_units'] = 256
    mc['depth'] = 3
    mc['attention_type'] = 'bahdanau'
    mc['embedding_size'] = 128
    mc['num_encoder_symbols'] = d.n_tokens
    mc['num_decoder_symbols'] = d.n_tokens
    mc['start_token'] = d.SOS
    mc['end_token'] = d.EOS
    mc['pad_token'] = d.PAD
    mc['use_residual'] = True
    mc['attn_input_feeding'] = False
    mc['use_dropout'] = False
    mc['dropout_rate'] = 0.0
    
    # Training
    mc['optimizer'] = 'adam'
    mc['learning_rate'] = 0.0002
    mc['max_gradient_norm'] = 1.0
    
    # Decoding
    mc['beam_width'] = 1
    mc['max_decode_step'] = 30
    return(mc)
    
