# -*- coding: utf8 -*-
"""Basic utilities and helper functions."""


import datetime
import pickle

from os import makedirs, walk
from os.path import isfile, exists, join, dirname, abspath


def get_path_files(path):
    """Returns file or files in a path.
    
    Args:
        path (str): File path to get files from.
        
    Returns:
        List of tiles, if provided path is a folder, otherwise file in a list.
    """
    if isfile(path):
        files = [path]
    else:
        files = list_files_in_folder(path)
    return files


def save_obj(obj,path,force=False):
    """Save object in pickle format.
    
    Args:
        path (str): Object save path.
        force (bool): Creates a folder, if it doesn't already exist.
    """
    if force:
        folder = dirname(path)
        create_folder(folder)
    with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
     
        
def load_obj(path):
    """Load pickled object.
    
    Args:
        path (str): Path to existing pickled object.
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return(obj)


def get_timestamp():
    """Get timestamp in a folder-name friendly format."""
    return(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def create_folder(folder):
    """Creates a folder it doesn't already exist."""
    if not exists(folder):
        makedirs(folder)
        print('Created folder %s!' % folder)
    
    
def get_abs_path(path):
    """Get provided path as absolute path.
    
    Args:
        path (str): Path to get absolute path into.
        
    Returns:
        Absolute path as a string.
    """
    absp = abspath(path)
    if absp[-1] not in '/\\':
        absp += '\\'
    return(absp)
    
    
def list_files_in_folder(folder):
    """Lists all files in a folder and its subfolders.
    
    Args:
        folder (str): Name of the folder
        
    Returns:
        List of files found in the folder and/or its subfolders.
    """ 
    files_list = []
    for path, _, files in walk(folder):
        for name in files:
            files_list.append(join(path, name))
    return(files_list)


def update_dict(original,replacer):
    """Updates a Python dict with another dict.
    
    Args:
        original (dict): Python dict to be updated.
        replacer (dict): Python dict to update the original dict with.
        
    Returns:
        The original dict updated with the replacer dict.
    """
    for k,v in replacer.items():
        original[k] = v
    return(original)
