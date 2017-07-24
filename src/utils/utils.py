# -*- coding: utf8 -*-
'''
Created on 12.7.2017

@author: Jesse
'''
import datetime
from os import listdir, makedirs, walk
from os.path import isfile, exists, join, dirname, abspath
import pickle


def save_obj(obj,filepath,force=False):
    if force:
        folder = dirname(filepath)
        create_folder(folder)
    with open(filepath, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return(obj)

def get_timestamp():
    return(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

def create_folder(folder):
    if not exists(folder):
        makedirs(folder)
        print('Created folder %s!' % folder)
    
def get_abs_path(path):
    absp = abspath(path)
    if absp[-1] not in '/\\':
        absp += '\\'
    return(absp)
    
def list_files_in_folder(folder):
    files_list = []
    for path, _, files in walk(folder):
        for name in files:
            files_list.append(join(path, name))
    return(files_list)

def update_dict(original,replacer):
    for k,v in replacer.items():
        original[k] = v
    return(original)