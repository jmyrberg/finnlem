# -*- coding: utf8 -*-
'''
Created on 24.7.2017

@author: Jesse
'''
import os
import string
import json

import numpy as np
import tensorflow as tf
from lemmer.lemmer import clean_tokens, Lemmer

from unidecode import unidecode
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer 

from models.seq2seq import Seq2Seq
from utils.utils import create_folder
from dictionary.dictionary import load_dict

def split_text_into_sentences(text):
    """Split text into sentences.
    
    Args:
        text (str): Text to be split into sentences.
        
    Returns:
        List of sentences.
    """
    return sent_tokenize(text)

def split_sentences_into_tokens(sentences):
    """Clean sentences for Titler class.
    
    Args:
        sentences: List of sentences.
        
    Returns:
        List of lists of tokens.
    """
    return [word_tokenize(sent) for sent in sentences]

class Titler(object):
    
    model_config = {}
    titler_config = {}
    
    def __init__(self, model_dir, lemmer_dir=None, 
                 dict_path=None, **kwargs):
        self.model_dir = model_dir
        self.lemmer_dir = lemmer_dir
        self.dict_path = dict_path
        self.model_config = kwargs
        
        self._init_config()
        
    def _init_config(self):
        """Initialize titler configurations."""
        self.config_path = os.path.join(self.model_dir,'model.config')
        if not os.path.exists(self.config_path):
            if self.lemmer_dir is None:
                raise ValueError("Lemmer directory needs to be provided!")
            if self.dict_path is None:
                raise ValueError("Dictionary path needs to be provided!")
            self._write_config()
        self._read_config()
        
    def _write_config(self):
        """Write titler and Seq2Seq -model configurations to file."""
        self.model_config['model_dir'] = self.model_dir
        self.titler_config['dict_path'] = self.dict_path
        self.titler_config['lemmer_dir'] = self.lemmer_dir
        config = {}
        config['model'] = self.model_config
        config['titler'] = self.titler_config
        create_folder(self.model_dir)
        with open(self.config_path,'w',encoding='utf8') as f:
            f.write(json.dumps(config,indent=2))
            
    def _read_config(self):
        """Read titler and Seq2Seq -model configurations from file."""
        with open(self.config_path,'r',encoding='utf8') as f:
                config = json.loads(f.read(),encoding='utf8')
        self.model_config = config['model']
        self.model_dir = config['model']['model_dir']
        self.titler_config = config['titler']
        self.lemmer_dir = config['titler']['lemmer_dir']
        self.dict_path = config['titler']['dict_path']
        self.dictionary = load_dict(self.dict_path)
        self.lemmer = SnowballStemmer('finnish')#Lemmer(model_dir=self.lemmer_dir)
        
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
        
    def _prepare_sentences(self,sentences):
        """
        Prepare sentences for training or decoding.
        
        Args: 
            sentences: List of sentences
          
        Returns: 
            <>: 
            <>: 
        """
        # Clean sentences without lemmer or dictionary
        sent_tokens = split_sentences_into_tokens(sentences)
        print('Original tokens:',sent_tokens)
                
        # Lemmatize and encode tokens
        lemmatized_sents = []
        for tokens in sent_tokens:
            
            lemmatized_tokens = [self.lemmer.stem(token) for token in tokens]
            
            # lemmatized_tokens: [n_tokens_in_sent[beam_width]]
            #lemmatized_tokens = self.lemmer.decode(tokens)
            # @TODO: Change this at some point [n_tokens_in_sent]
            # lemmatized_tokens: [n_tokens_in_sent]
            #lemmatized_tokens = [beam[0] for beam in lemmatized_tokens]
            
            # lemmatized_sents: [n_sents[n_tokens_in_sent]]
            lemmatized_sents.append(lemmatized_tokens)
            
        print('Lemmatized sents:',lemmatized_sents)

        # Encode tokens using dictionary
        encoded_seqs = self.dictionary.docs2seqs(lemmatized_sents,
                                                 return_length=True)
        seqs,seq_lens = zip(*list(encoded_seqs))
        seq_lens = np.array(seq_lens, dtype=np.int32).flatten()
        
        # Pad sequences
        maxlen = seq_lens.max()
        seqs = pad_sequences(seqs, maxlen=maxlen, dtype=np.int32,
                             padding='post', truncating='post',
                             value=self.dictionary.PAD)
        
        return seqs,seq_lens
         
        
    def _convert_seqs_to_sentences(self,seqs):
        """Convert decoded sequences back to sentences
        
        Args:
            seqs: Array of sequences, size 
                  [batch_size,max_decode_steps,beam_width]
        
        Returns:
            Lists of lists of decoded sentences, with length batch_size
        """
        decoded_sents = []
        for seq in seqs:
            decoded_sents_beams = []
            for k in range(seq.shape[1]):
                sent_tokens = self.dictionary.seq2doc(seq[:,k])
                decoded_sent = " ".join(sent_tokens)
                decoded_sents_beams.append(decoded_sent)
                print(decoded_sent)
            decoded_sents.append(decoded_sents_beams)
        return decoded_sents
        
    def train(self,source_sentences,target_sentences):
        """Perform a training step.
        
        Args:
            source_tokens: List of tokens to feed in encoder
            target_tokens: list of tokens to use as a decoder target
            
        Returns:
            Loss after training step has been performed
        """
        self._set_model('train')
        source_seqs,source_lens = self._prepare_sentences(source_sentences)
        target_seqs,target_lens = self._prepare_sentences(target_sentences)
        loss,global_step = self.model.train(source_seqs,source_lens,
                                 target_seqs,target_lens)
        if global_step % 1 == 0:
            self.model.save()
        return loss
        
    def decode(self,source_sentences):
        """Decode a list of tokens into their baseform.
        
        Args:
            source_tokens: List of tokens to feed in encoder
            
        Returns:
            List of tokens in their baseform
        """
        self._set_model('decode')
        seqs,seq_lens = self._prepare_sentences(source_sentences)
        pred_seqs = self.model.decode(seqs,seq_lens)
        decoded_tokens = self._convert_seqs_to_sentences(pred_seqs)
        return decoded_tokens
        
titler = Titler(model_dir='../data/titler/models/testi',
                lemmer_dir='../data/lemmer/models/testi',
                dict_path='../data/titler/dicts/dictionary.dict')
for i in range(1000):
    loss = titler.train(['opuksemme rollerisi']*16,['rollerisi opusksemme']*16)
    print(loss)
    if i % 10 == 0:
        print(titler.decode(['opuksemme rollerisi']))