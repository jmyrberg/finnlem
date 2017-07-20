# -*- coding: utf8 -*-
'''
Created on 11.7.2017

@author: Jesse
'''
import re
import string
from nltk import word_tokenize
from nltk.stem.snowball import FinnishStemmer

class Preprocessor():
    
    def __init__(self,
                 tokenize=True,
                 remove_punctuation=False,
                 stem=True,
                 smart_lower=True,
                 ignore_stopwords=True,
                 minlen=2):
        self.tokenize = tokenize
        self.remove_punctuation = remove_punctuation
        self.stem = stem
        self.smart_lower = smart_lower
        self.ignore_stopwords = ignore_stopwords
        self.minlen = minlen
        
        self._init_all()
        
    def _init_all(self):
        if self.stem:
            self.stemmer = FinnishStemmer(ignore_stopwords=self.ignore_stopwords)
        
    def _tokenize(self,doc):
        doc = word_tokenize(doc, language='finnish')
        return(doc)
    
    def _remove_punctuation(self,doc):
        doc = [token for token in doc if token not in string.punctuation]
        return(doc)
    
    def _smart_lower(self,doc):
        previous_token = '.'
        new_doc = []
        for token in doc:
            if previous_token == '.':
                token = token.lower()
            new_doc.append(token)
            previous_token = token
        return(new_doc)
    
    def _stem(self,doc):
        new_doc = []
        for token in doc:
            if token.islower():
                new_doc.append(self.stemmer.stem(token))
            else:
                new_doc.append(token)
        return(new_doc)
    
    def _remove_short(self,doc):
        if self.remove_punctuation:
            doc = [token for token in doc if len(token) >= self.minlen]
        else:
            doc = [token for token in doc if len(token) >= self.minlen and token not in string.punctuation]
        return(doc)
    
    def process_doc(self,doc):
        if self.tokenize:
            doc = self._tokenize(doc)
        if self.stem:
            doc = self._stem(doc)
        if self.smart_lower:
            doc = self._smart_lower(doc)
        if self.remove_punctuation:
            doc = self._remove_punctuation(doc)
        doc = self._remove_short(doc)
        return(doc)
        
    def process_docs(self,docs):
        for doc in docs:
            yield self.process_doc(doc)
        