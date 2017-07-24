# -*- coding:utf8 -*-
'''
Created on 24.7.2017

@author: Jesse Myrberg
'''
import os
import string
import json

import numpy as np
import tensorflow as tf

from unidecode import unidecode
from keras.preprocessing.sequence import pad_sequences

from models.seq2seq import Seq2Seq
from utils.utils import create_folder
from dictionary.dictionary import load_dict

ALLOWED_ACCENTS = ['ä','ö']
        
MIN_LEN = 3
MAX_LEN = 40  

LEN_REPLACE_CHAR = '<UNK>'
NUM_REPLACE_CHAR = 'num'

def clean_tokens(tokens):
    """Clean tokens for Lemmer class.
    
    Args:
        tokens: List of tokens.
        
    Returns:
        List of cleaned tokens.
    """
    
    # Clean up tokens
    cleaned_tokens = []
    for token in tokens:
        
        # Remove punctuation
        token = "".join([char for char in token 
                         if char not in string.punctuation])
        
        # Token is a number
        if token.isdigit():
            cleaned_tokens.append(NUM_REPLACE_CHAR)
            continue
        
        # Token is too short or long
        n_token = len(token)
        if n_token < MIN_LEN or n_token > MAX_LEN:
            cleaned_tokens.append(LEN_REPLACE_CHAR)
            continue
        
        # Normalize accents
        tmp_token = ''
        for char in token:
            if char not in ALLOWED_ACCENTS:
                norm_char = unidecode(char)
                tmp_token += norm_char
            else:
                tmp_token += char
        token = tmp_token
        
        # Lowercase and strip
        token = token.lower().strip()
        
        cleaned_tokens.append(token)
        
    return cleaned_tokens


class Lemmer(object):
    """Wrapper for Seq2Seq -class from models.seq2seq.
    
    Args:
        model_dir (str): Directory in which the Lemmer exists, or will 
            be created if it is initiated for the first time.
        dict_path (str): Path to dictionary to be used in the model.
            This needs to be provided, when creating the model for 
            the first time. Defaults to None.
        **kwargs: Arbitrary keyword arguments for Seq2Seq class.
        
    Attributes:
        <None>
        
    Examples:
        Create Lemmer for the first time:
        >>> lemmer = Lemmer(model_dir,dict_path)
        
        Load existing Lemmer:
        >>> lemmer = Lemmer(model_dir)
        
        Train lemmer with tokens:
        >>> source_tokens = ['koiramme']*16
        >>> target_tokens = ['koira']*16
        >>> loss = lemmer.train(source_tokens,target_tokens)
        
        Loss after training step:
        >>> print(loss)
        
        >>> # Decode tokens
        >>> tokens_to_decode = ['koiramme']
        >>> decoded_tokens = lemmer.decode(tokens_to_decode)
        >>> print(decoded_tokens)
    
    """
    
    lemmer_config = {}
    model_config = {}
    
    def __init__(self, model_dir, dict_path=None, **kwargs):
        self.model_dir = model_dir
        self.dict_path = dict_path
        self.model_config = kwargs
        
        self._init_config()
        
    def _init_config(self):
        """Initialize lemmer configurations."""
        self.config_path = os.path.join(self.model_dir,'model.config')
        # Model doesn't exist
        if not os.path.exists(self.config_path):
            # No dictionary path provided
            if self.dict_path is None:
                raise ValueError("Dictionary path needs to be provided!")
            else:
                self._write_config()
        self._read_config()
        
    def _write_config(self):
        """Write Lemmer and Seq2Seq -model configurations to file."""
        self.model_config['model_dir'] = self.model_dir
        self.lemmer_config['dict_path'] = self.dict_path
        config = {}
        config['model'] = self.model_config
        config['lemmer'] = self.lemmer_config
        create_folder(self.model_dir)
        with open(self.config_path,'w',encoding='utf8') as f:
            f.write(json.dumps(config,indent=2))
            
    def _read_config(self):
        """Read Lemmer and Seq2Seq -model configurations from file."""
        with open(self.config_path,'r',encoding='utf8') as f:
                config = json.loads(f.read(),encoding='utf8')
        self.model_config = config['model']
        self.model_dir = config['model']['model_dir']
        self.lemmer_config = config['lemmer']
        self.dict_path = config['lemmer'].get('dict_path')
        self.dictionary = load_dict(self.dict_path)
        
    def _set_model(self,mode):
        """Change the mode of Seq2Seq -model."""
        if hasattr(self,'model') and self.model.mode == mode:
            pass
        elif hasattr(self,'model') and not self.model.mode == mode:
            self.model.sess.close()
            del self.model
            self._set_model(mode)
        elif not hasattr(self,'model'):
            model_config = self.model_config
            model_config['mode'] = mode
            self.model = Seq2Seq(**model_config)
            print('Model mode changed to',mode)
        
    def _prepare_tokens(self,tokens):
        """
        Prepare tokens for training or decoding.
        
        Args: 
            tokens: List of tokens
          
        Returns: 
            seqs: Dictionary-encoded token sequences
            seq_lens: Sequence lengths without padding included
        """
        # Clean tokens without dictionary
        cleaned_tokens = clean_tokens(tokens)
        
        # Encode tokens using dictionary
        seqs = self.dictionary.docs2seqs(cleaned_tokens,
                                         return_length=True)
        seqs,seq_lens = zip(*seqs)
        seq_lens = np.array(seq_lens, dtype=np.int32).flatten()
        
        # Pad sequences
        maxlen = seq_lens.max()
        seqs = pad_sequences(seqs, maxlen=maxlen, dtype=np.int32,
                             padding='post', truncating='post',
                             value=self.dictionary.PAD)
        
        return seqs,seq_lens
         
        
    def _convert_seqs_to_tokens(self,seqs):
        """Convert decoded sequences back to tokens
        
        Args:
            seqs: Array of sequences, size 
                  [batch_size,max_decode_steps,beam_width]
        
        Returns:
            Lists of lists of decoded tokens, with length batch_size
        """
        decoded_tokens = []
        for seq in seqs:
            decoded_token_beams = []
            for k in range(seq.shape[1]):
                token_chars = self.dictionary.seq2doc(seq[:,k])
                decoded_token = "".join(token_chars)
                decoded_token_beams.append(decoded_token)
            decoded_tokens.append(decoded_token_beams)
        return decoded_tokens
        
    def train(self,source_tokens,target_tokens):
        """Perform a training step.
        
        Args:
            source_tokens: List of tokens to feed in encoder
            target_tokens: list of tokens to use as a decoder target
            
        Returns:
            Loss after training step has been performed
        """
        self._set_model('train')
        source_seqs,source_lens = self._prepare_tokens(source_tokens)
        target_seqs,target_lens = self._prepare_tokens(target_tokens)
        loss,global_step = self.model.train(source_seqs,source_lens,
                                 target_seqs,target_lens)
        if global_step % 100 == 0:
            self.model.save()
        return loss
        
    def decode(self,source_tokens):
        """Decode a list of tokens into their baseform.
        
        Args:
            source_tokens: List of tokens to feed in encoder
            
        Returns:
            List of tokens in their baseform
        """
        self._set_model('decode')
        seqs,seq_lens = self._prepare_tokens(source_tokens)
        pred_seqs = self.model.decode(seqs,seq_lens)
        decoded_tokens = self._convert_seqs_to_tokens(pred_seqs)
        return decoded_tokens
    
    
# l = Lemmer('../data/lemmer/models/testi')
# for i in range(100):
#     loss = l.train(['opuksemme','rollerisi']*32,['opus','rolleri']*32)
#     print(i,loss)
# print(l.decode(['opuksemme.','rollerisi']))