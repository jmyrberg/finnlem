# -*- coding: utf8 -*-
'''
Created on 12.7.2017

@author: Jesse
'''
import numpy as np
from utils import save_obj,load_obj
from collections import defaultdict, Counter
from model_processing import Preprocessor

class Dictionary(object):
    
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
    
    def __init__(self, 
                 preprocessor=Preprocessor(),
                 vocab_size=100000,
                 min_freq=0.1,
                 max_freq=1.0,
                 prune_every_n=5000):
        self.preprocessor = preprocessor
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.prune_every_n = prune_every_n
        
        self._init_dicts()
        
    def _init_dicts(self,already_init=False):
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
        print('Dictionary info')
        print('Number of tokens:',self.n_tokens)
        print('Number of tokens in token2id:',len(self.token2id))
        print('Number of tokens in id2token:',len(self.id2token))
        print('Number of tokens in counter:',len(self.counter))
        print('Counter top 10:',self.counter.most_common(10))
        
    def _try_remove_token(self,token,count,reset_counter=True):
        if token in self.token2id and token not in self.special_tokens:
            del self.id2token[self.token2id[token]]
            del self.token2id[token]
            self.n_tokens -= 1
            
            if reset_counter:
                self.token2UNK.update([token])
                del self.counter[token]
                self.counter['UNK'] += count
            
    def _keep_n(self,reset_counter):
        if self.n_tokens > self.vocab_size:
            to_remove = self.counter.most_common()
            to_remove = [(token,count) for token,count in to_remove if token not in self.special_tokens]
            to_remove = to_remove[-(len(to_remove)-self.vocab_size+self.n_special_tokens):]
            for token,count in to_remove:
                self._try_remove_token(token,count,reset_counter)
        
    def _remove_extremes(self):
        tokens_counts = list(self.counter.items())
        _,counts = zip(*tokens_counts)
        min_count = np.percentile(counts, q=self.min_freq*100) - 1e-6
        max_count = np.percentile(counts, q=self.max_freq*100) + 1e-6
        for token,count in tokens_counts:
            if count <= min_count or count >= max_count:
                self._try_remove_token(token,count)
            
    def _compactify(self):
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
        
    def fit_batch(self,docs,verbose=True):
        for doc in self.preprocessor.process_docs(docs):
            self.n_docs += 1
            for token in doc:
                if token not in self.token2id:
                    self.id2token[self.n_tokens] = token
                    self.token2id[token] = self.n_tokens
                    self.n_tokens += 1
                self.counter[token] += 1
                
            if self.n_docs % self.prune_every_n == 0:
                self._keep_n(reset_counter=False)
                self._compactify()
                
        if verbose:
            print('Dictionary fitted with %d documents (%d tokens)' % \
                  (self.n_docs,self.n_tokens))
        
    def lock(self):
        self._remove_extremes()
        self._keep_n(reset_counter=True)
        self._compactify()
        self.locked = True
        print('Dictionary fitted with %d documents (%d tokens)' % \
                  (self.n_docs,self.n_tokens))
        print('Dictionary locked!')
        
    def seq2doc(self,seq):
        doc = [self.id2token[i] for i in seq]
        return(doc)
    
    def seqs2docs(self,seqs):
        for seq in seqs:
            yield self.seq2doc(seq)
    
    def doc2seq(self,doc,prepend_SOS=False,append_EOS=False):
        doc = self.preprocessor.process_doc(doc)
        cnt = 0
        seq = np.zeros(len(doc)+prepend_SOS+append_EOS,dtype=np.int32)
        if prepend_SOS:
            seq[0] = self.SOS
            cnt += 1
        for i,token in enumerate(doc):
            seq[i+cnt] = self.token2id.get(token,self.UNK)
        if append_EOS:
            seq[i+cnt+1] = self.EOS
        return(seq)
    
    def docs2seqs(self,docs,prepend_GO=False,append_EOS=False):
        for doc in docs:
            yield self.doc2seq(doc,
                               prepend_GO=prepend_GO,
                               append_EOS=append_EOS)
    
    def save(self,save_path):
        if self.locked:
            save_obj(self,save_path)
            save_obj(self.counter,save_path+'.counter')
            self.path = save_path
        
    def load(self,load_path):
        dictionary = load_obj(load_path)
        dictionary.counter = load_obj(load_path+'.counter')
        dictionary.path = load_path
        return(dictionary)
    