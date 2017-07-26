# -*- coding: utf8 -*-
"""Wrappers for tensorflow models."""


import os
import json

import numpy as np
import tensorflow as tf

from unidecode import unidecode
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from nltk.stem import SnowballStemmer 

from tf_models import Seq2SeqModel
from utils import create_folder
from dictionary import load_dict


ALLOWED_ACCENTS = ['ä','ö']
STEMMER = SnowballStemmer('finnish')
DETOKENIZER = MosesDetokenizer('finnish')
"""
Module level variables:
    ALLOWED_ACCENTS (list): List of accented letters that shouldn't be removed.
    STEMMER (SnowballStemmer): Stemmer to use for stemming. Must have a 
        'stem' -method that takes a token as an input.
    DETOKENIER (MosesDetokenizer): Detokenizer to use for constructing
        the original string back. Must have a 'detokenize' -method.
"""


def doc_to_tokens(doc):
    """Convert a document string to tokens.
    
    Args:
        doc (str): Document to split into tokens.
        
    Returns:
        List of tokens.
        
    Example:
        >>> doc = 'This is my sentence.'
        >>> tokens = doc_to_tokens(doc)
        >>> print(tokens)
    """
    try:
        # Tokenize
        tokens = word_tokenize(doc)
        tokens_stemmed = [STEMMER.stem(token) for token in tokens]
        
        # Remove accents
        cleaned_tokens = []
        for token in tokens_stemmed:
            cleaned_token = ''
            for char in token:
                if char not in ALLOWED_ACCENTS:
                    cleaned_token += unidecode(char)
                else:
                    cleaned_token += char
            cleaned_tokens.append(token)    
    except:
        cleaned_tokens = ['<UNK>']
    finally:   
        return cleaned_tokens


class Seq2Seq(object):
    """Wrapper for Seq2SeqModel.
    
    Args:
        model_dir (str): 
        dict_path (str): 
        **kwargs: 
        
    Examples:
        Create a new model:
        >>> # Path to folder where the model will be created
        >>> model_dir = './MyModel/'
        >>> # Path to existing, trained Dictionary
        >>> dict_path = './data/dicts/dictionary.dict' 
        >>> # Create model with default params
        >>> m = Seq2Seq(model_dir, dict_path)
        
        Train a batch:
        >>> source_docs = ['koiramme','koirasi']
        >>> target_docs = ['koira','koira']
        >>> m.train(source_docs, target_docs)
        
        Decode a batch:
        >>> decoded_docs = m.decode(['koirasi','koiramme'])
        >>> print(decoded_docs)
    """
    
    model_config = {}
    titler_config = {}
    
    def __init__(self, model_dir, dict_path=None, **kwargs):
        self.model_dir = model_dir
        self.dict_path = dict_path
        
        self.model_config = kwargs
        
        self._init_config()
        
    def _init_config(self):
        """Initialize model configurations."""
        self.config_path = os.path.join(self.model_dir,'model.config')
        if not os.path.exists(self.config_path):
            if self.dict_path is None:
                raise ValueError("Dictionary path needs to be provided!")
            self._write_config()
        self._read_config()
        
    def _write_config(self):
        """Write model configuration file."""
        self.model_config['model_dir'] = self.model_dir
        self.titler_config['dict_path'] = self.dict_path
        
        config = {}
        config['model'] = self.model_config
        config['titler'] = self.titler_config
        
        create_folder(self.model_dir)
        with open(self.config_path,'w',encoding='utf8') as f:
            f.write(json.dumps(config,indent=2))
            
    def _read_config(self):
        """Read model configurations from file."""
        with open(self.config_path,'r',encoding='utf8') as f:
                config = json.loads(f.read(),encoding='utf8')
        
        self.model_config = config['model']
        self.model_dir = config['model']['model_dir']
        
        self.titler_config = config['titler']
        
        self.dict_path = config['titler']['dict_path']
        self.dictionary = load_dict(self.dict_path)
        
    def _set_model(self ,mode):
        """Toggle between 'train' and 'decode mode of Seq2Seq."""
        if hasattr(self,'model') and self.model.mode == mode:
            pass
        elif hasattr(self,'model') and not self.model.mode == mode:
            self.model.sess.close()
            tf.reset_default_graph()
            del self.model
            self._set_model(mode)
        elif not hasattr(self,'model'):
            self._create_model(mode)
        
    def _create_model(self, mode):
        """Create a new Seq2SeqModel."""
        model_config = self.model_config
        model_config['mode'] = mode
        model_config['num_encoder_symbols'] = self.dictionary.n_tokens
        model_config['num_decoder_symbols'] = self.dictionary.n_tokens
        model_config['start_token'] = self.dictionary.SOS
        model_config['end_token'] = self.dictionary.EOS
        model_config['pad_token'] = self.dictionary.PAD
        self.model = Seq2SeqModel(**model_config)
        self.model.sess.graph.finalize()
        
    def _docs_to_seqs(self, docs, max_seq_len=None):
        """Convert documents to sequences.
        
        Args: 
            docs (list): List of document strings.
            max_seq_len (int): Maximum sequence length. Defaults to None. 
          
        Returns:
            Padded array of sequences and their lengths without padding.
        """
        # Documents to tokens
        tokens = [doc_to_tokens(doc) for doc in docs]

        # Encode tokens using dictionary_
        encoded_seqs = self.dictionary.docs2seqs(tokens,
                                                 return_length=True)
        seqs,seq_lens = zip(*list(encoded_seqs))
        seq_lens = np.array(seq_lens, dtype=np.int32).flatten()
        
        # Pad sequences
        if max_seq_len is None:
            maxlen = seq_lens.max()
        else:
            maxlen = min((seq_lens.max(),max_seq_len))
        seqs = pad_sequences(seqs, maxlen=maxlen, dtype=np.int32,
                             padding='post', truncating='post',
                             value=self.dictionary.PAD)
        return seqs,seq_lens
        
    def _seqs_to_docs(self, seqs):
        """Convert decoded sequences back to sentences
        
        Args:
            seqs: Array of sequences, with the size of
                [batch_size,max_decode_steps,beam_width]
        
        Returns:
            List of lists of decoded sentences.
        """
        docs = []
        for seq in seqs:
            beams = []
            for k in range(seq.shape[1]):
                tokens = self.dictionary.seq2doc(seq[:,k])
                detokenized = DETOKENIZER.detokenize(tokens,return_str=True)
                beams.append(detokenized)
            docs.append(beams)
        return docs
        
    def train(self, source_docs, target_docs, max_seq_len=None):
        """Perform a model training step.
        
        Args:
            source_docs: List of source documents to feed in encoder.
            target_docs: List of target documents for decoder to learn from.
            
        Returns:
            Bathc loss and global step of the model.
        """
        self._set_model('train')
        source_seqs,source_lens = self._docs_to_seqs(source_docs,
                                                     max_seq_len)
        target_seqs,target_lens = self._docs_to_seqs(target_docs,
                                                     max_seq_len)
        loss,global_step = self.model.train(source_seqs,source_lens,
                                            target_seqs,target_lens)
        if global_step % 512 == 0:
            self.model.save()
            print('Model saved!')
        return loss,global_step
        
    def decode(self, source_docs):
        """Decode source documents to their target form.
        
        Args:
            source_docs: List of documents to feed in encoder.
            
        Returns:
            List of decoded documents.
        """
        self._set_model('decode')
        seqs,seq_lens = self._docs_to_seqs(source_docs)
        pred_seqs = self.model.decode(seqs,seq_lens)
        decoded_docs = self._seqs_to_docs(pred_seqs)
        return decoded_docs
        