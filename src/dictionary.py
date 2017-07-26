# -*- coding: utf8 -*-
"""Dictionary class that handles vocabulary mappings."""


import os

from collections import defaultdict, Counter

import numpy as np

from utils import save_obj, load_obj


def load_dict(load_path):
    """Load existing Dictionary object.
    
    Args:
        load_path (str): Path to existing dictionary.
        
    Returns:
        Dictionary object
        
    Raises:
        FileNotFoundError: If dictionary or counter path doesn't exist.
    """
    counter_path = load_path + '.counter'
    if not os.path.exists(load_path):
        raise FileNotFoundError("Dictionary path %s doesn't exist" % \
                                load_path)
    if not os.path.exists(load_path+'.counter'):
        raise FileNotFoundError("Dictionary counter %s doesn't exist" % \
                                counter_path)
    dictionary = load_obj(load_path)
    dictionary.counter = load_obj(load_path+'.counter')
    dictionary.path = load_path
    return(dictionary)
    
    
def save_dict(dictionary,save_path):
    """Save Dictionary object to file.
    
    Args:
        dictionary (Dictionary class): Dictionary to save.
        save_path (str): Dictionary save path. Counter will be saved to
            to path 'save_path'+'.counter'.
        
    Raises:
        ValueError: If Dictionary object is not locked.
    """
    if dictionary.locked:
        save_obj(dictionary,save_path,force=True)
        save_obj(dictionary.counter,save_path+'.counter',force=True)
    else:
        raise ValueError("Dictionary must be locked before saving. " + \
                         "Try using the .lock() -method.")


class Dictionary(object):
    """Vocabulary dictionary with special tokens for seq-to-seq modelling.
    
    Args:
        vocab_size (int): Size of vocabulary, including special tokens
        min_freq (float): Keep words that exceed this minimum word frequency 
            in training data. Float between 0 and 1. Defaults to 0.
        max_freq (float): Keep words below this maximum word frequency in
            training data. Float between 0 and 1. Defaults to 1.
            
    Attributes:
        id2token (dict): Dictionary of id (int) -> token (str) mappings.
        token2id (dict): Dictionary of token (str) -> id (int) mappings.
        counter (Counter): Dictionary of token counts. For most common tokens,
            use 'Dictionary.counter.most_common()'.
        n_docs (int): Number of documents the Dictionary has been fitted on.
        n_tokens (int): Number of tokens.
        special_tokens (list): List of special tokens.
        n_special_tokens (int): Number of special tokens.
        SOS: Integer representation for start of sequence token
        PAD: Integer representation for padding token
        EOS: Integer representation for end of sequence token
        UNK: Integer representation for unknown token
    
    Examples:
        Create a new Dictionary:
        >>> from dictionary import Dictionary
        >>> d = Dictionary(10000, 0.1, 1.0)
        
        Fit documents in batches:
        >>> docs = [['This','is','a','document','.']*16]
        >>> d.fit_batch(docs)
        >>> d.fit_batch(docs) # Fit another batch
        >>> print('Number of tokens:',d.n_tokens)
        >>> d.lock() # Lock dictionary after batch fitting in order to save
        
        Fit documents in one batch:
        >>> d = Dictionary(10000, 0.1, 1.0)
        >>> d.fit(docs*2) # This automatically locks the dictionary
        
        Save the fitted Dictionary:
        >>> d.save('MyDictionary.dict') # or using save_dict -function
        
        Load existing Dictionary:
        >>> from dictionary import load_dict
        >>> d = load_dict('MyDictionary.dict')
        >>> print('Number of tokens after loading:',d.n_tokens)
        
        Map documents to id sequences:
        >>> id_seqs = list(d.docs2seqs(docs)) # From generator to list
        >>> print(id_seqs)
        
        Map id sequences back to documents
        >>> docs_converted = list(d.seqs2docs(id_seqs))
        >>> print(docs_converted)
    """
    
    locked = False
    id2token = defaultdict(str)
    token2id = defaultdict(int)
    token2UNK = set()
    counter = Counter()
    n_docs = 0
    n_special_tokens = 4
    n_tokens = 4
    SOS=2
    PAD=0
    EOS=1
    UNK=3
    special_tokens = ['<SOS>','<PAD>','<EOS>','<UNK>']
    path = None
    
    def __init__(self, vocab_size=100000, min_freq=0.1, max_freq=1.0):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        self._init_dicts()
        
    def _init_dicts(self, already_init=False):
        """Initialize special tokens."""
        self.id2token[self.PAD] = '<PAD>'
        self.id2token[self.EOS] = '<EOS>'
        self.id2token[self.UNK] = '<UNK>'
        self.id2token[self.SOS] = '<SOS>'
        self.token2id['<PAD>'] = self.PAD
        self.token2id['<EOS>'] = self.EOS
        self.token2id['<UNK>'] = self.UNK
        self.token2id['<SOS>'] = self.SOS
        if not already_init:
            for token in self.special_tokens:
                self.counter[token] = 0
        
    def _print_debug(self):
        """Print info about Dictionary."""
        print('Dictionary info')
        print('Number of tokens:',self.n_tokens)
        print('Number of tokens in token2id:',len(self.token2id))
        print('Number of tokens in id2token:',len(self.id2token))
        print('Number of tokens in counter:',len(self.counter))
        print('Counter top 10:',self.counter.most_common(10))
        
    def _try_remove_token(self, token, count, reset_counter=True):
        """Try to remove a token from dictionary."""
        if token in self.token2id and token not in self.special_tokens:
            del self.id2token[self.token2id[token]]
            del self.token2id[token]
            self.n_tokens -= 1
            
            if reset_counter:
                self.token2UNK.update([token])
                del self.counter[token]
                self.counter['UNK'] += count
            
    def _keep_n(self, reset_counter):
        """Filter Dictionary to have n most common tokens."""
        if self.n_tokens > self.vocab_size:
            to_remove = self.counter.most_common()
            to_remove = [(token,count) for token,count in to_remove 
                         if token not in self.special_tokens]
            start_idx = len(to_remove) - self.vocab_size + self.n_special_tokens
            to_remove = to_remove[-start_idx:]
            for token,count in to_remove:
                self._try_remove_token(token,count,reset_counter)
        
    def _remove_extremes(self):
        """Filter Dictionary based on min and max frequencies."""
        tokens_counts = list(self.counter.items())
        _,counts = zip(*tokens_counts)
        min_count = np.percentile(counts, q=self.min_freq*100) - 1e-6
        max_count = np.percentile(counts, q=self.max_freq*100) + 1e-6
        for token,count in tokens_counts:
            if count <= min_count or count >= max_count:
                self._try_remove_token(token,count)
            
    def _compactify(self):
        """Prune id's to start from zero."""
        n_tokens = self.n_special_tokens
        id2token = defaultdict(str)
        token2id = defaultdict(int)
        for token in self.token2id.keys():
            if token not in self.special_tokens:
                id2token[n_tokens] = token
                token2id[token] = n_tokens
                n_tokens += 1
        self.n_tokens = n_tokens
        self.id2token = id2token
        self.token2id = token2id
        self._init_dicts(already_init=True)
        
    def fit_batch(self, docs, prune_every_n=None):
        """Fit a batch of documents in Dictionary.
        
        Note:
            Use .lock() -method after batches have been fit.
        
        Args:
            docs (list): List of lists of tokens. For example,
                [['This','is','first','.'],['This','is','second','.']]
            prune_every_n (int): Every n batch, filter vocab_size most common 
                tokens, and start id's from zero again. This speeds up 
                training. Defaults to None.
                
        Example:
            Fit documents in batches:
            >>> docs = [['This','is','a','document','.']*16]
            >>> d = Dictionary()
            >>> d.fit_batch(docs)
            >>> d.fit_batch(docs) # Fit another batch
            >>> print('Number of tokens:',d.n_tokens)
            >>> d.lock() # Lock dictionary after batch fitting in order to save
        """
        for doc in docs:
            self.n_docs += 1
            for token in doc:
                if token not in self.token2id:
                    self.id2token[self.n_tokens] = token
                    self.token2id[token] = self.n_tokens
                    self.n_tokens += 1
                self.counter[token] += 1
                
            if prune_every_n is not None:
                if self.n_docs % prune_every_n == 0:
                    self._keep_n(reset_counter=False)
                    self._compactify()
                    print('\nNumber of documents:',self.n_docs)
                    self._print_debug()
        
    def fit(self, docs):
        """Fit documents in Dictionary.
        
        Args:
            docs (list): List of lists of tokens. For example,
                [['This','is','first','.'],['This','is','second','.']]
                
        Example:
            Fit documents in batches:
            >>> docs = [['This','is','a','document','.']*16]
            >>> d = Dictionary()
            >>> d.fit(docs)
        """
        self.fit_batch(docs)
        self.lock()
        
    def lock(self):
        """Lock-in vocabulary to be able to save."""
        self._remove_extremes()
        self._keep_n(reset_counter=True)
        self._compactify()
        self.locked = True
        print('Dictionary fitted with %d documents (%d tokens)' % \
              (self.n_docs,self.n_tokens))
        
    def seq2doc(self, seq, remove_EOS=True):
        """Map id sequence to document.
        
        Args:
            seq (list): List of ids.
            remove_EOS (boolean): Remove end of sequence id from the end 
                of the sequence. Defaults to True.
                
        Returns:
            List of tokens.
        """
        doc = []
        for i in seq:
            if remove_EOS and i == self.EOS:
                break
            else:
                doc.append(self.id2token[i])
        return(doc)
    
    def seqs2docs(self, seqs):
        """Map id sequences to documents.
        
        Args:
            seqs (list): List of lists of id sequences.
                
        Yields:
            List of tokens, one at a time.
        """
        for seq in seqs:
            yield self.seq2doc(seq)
    
    def doc2seq(self,doc,return_length=False,
                prepend_SOS=False,append_EOS=False):
        """Map document to id sequence.
        
        Args:
            doc (list): List of tokens.
            return_length (bool): Return the length of sequence, after
                mapping tokens to ids. Defaults to False.
            prepend_SOS (bool): Prepend start of sequence id at the start 
                of sequence. Defaults to False.
            append_EOS (bool): Append end of sequence id at the end of 
                sequence. Defaults to False.
                
        Returns:
            List of ids, if return_length=False.
            Length and list of ids, if return_length=True.
        """
        cnt = 0
        seq_len = len(doc)+prepend_SOS+append_EOS
        seq = np.zeros(seq_len,dtype=np.int32)
        if prepend_SOS:
            seq[0] = self.SOS
            cnt += 1
        for i,token in enumerate(doc):
            seq[i+cnt] = self.token2id.get(token,self.UNK)
        if append_EOS:
            seq[i+cnt+1] = self.EOS
        if not return_length:
            return seq
        if return_length:
            return seq,[seq_len]
    
    def docs2seqs(self, docs, return_length=False,
                  prepend_SOS=False, append_EOS=False):
        """Map documents to id sequences.
        
        Args:
            docs (list): List of lists of tokens.
            return_length (bool): Return the length of sequence, after
                mapping tokens to ids. Defaults to False.
            prepend_SOS (bool): Prepend start of sequence id at the start 
                of sequence. Defaults to False.
            append_EOS (bool): Append end of sequence id at the end of 
                sequence. Defaults to False.
                
        Yields:
            List of ids, if return_length=False. One doc at a time.
            Length and list of ids, if return_length=True. One doc at a time.
        """
        for doc in docs:
            yield self.doc2seq(doc,
                               return_length=return_length,
                               prepend_SOS=prepend_SOS,
                               append_EOS=append_EOS)
    
    def save(self,save_path):
        """Save Dictionary object to file.
    
        Args:
            dictionary (Dictionary class): Dictionary to save.
            save_path (str): Dictionary save path. Counter will be saved to
                to path 'save_path'+'.counter'.
            
        Raises:
            ValueError: If Dictionary object is not locked.
        """
        save_dict(self,save_path)
    