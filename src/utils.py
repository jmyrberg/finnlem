# -*- coding: utf8 -*-
'''
Created on 12.7.2017

@author: Jesse
'''
import pickle
import datetime
from os.path import isfile,exists,join,dirname,abspath
from os import listdir,makedirs,walk

def save_obj(obj,filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print('Saved successfully in %s!' % filename)
        
def load_obj(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print('Object loaded successfully from %s!' % filename)
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