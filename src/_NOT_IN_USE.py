# -*- coding: utf8 -*-
'''
Created on 12.7.2017

@author: Jesse
'''
import json
import csv
import re
from utils import list_files_in_folder, create_folder

class FeedProcessor(object):
    
    lines_per_file = 1024
    
    def __init__(self, in_folder, out_folder, process_func):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.process_func = process_func
        
        self._init_filenames()
        
    def _init_filenames(self):
        self.in_files = list_files_in_folder(self.in_folder)
        if self.out_folder[-1] not in '/\\':
            self.out_folder += '/'
        self.out_files = [self.out_folder+f.split('\\')[-1] \
                          for f in self.in_files]
        print('%d number of files found' % len(self.in_files))
        
    def _read_jsonlines_as_dict(self,filename):
        with open(filename,'r',encoding='utf8') as f:
            for line in f:
                d = json.loads(line)
                yield d
                
    def process(self):
        create_folder(self.out_folder)
        for in_filename,out_filename in zip(self.in_files,self.out_files):
            
            nb_lines_written = 0
            nb_files_written = 0
            out_filename_str = out_filename+'-'+str(nb_files_written).zfill(4)
            f = open(out_filename_str,'w',encoding='utf8')
            writer = csv.DictWriter(f, fieldnames=['input','target'])
            writer.writeheader()
            print('Processing',in_filename,'-->',out_filename_str)
            
            for d in self._read_jsonlines_as_dict(in_filename):
                input_doc,target_doc = self.process_func(d)
                writer.writerow({'input':input_doc,
                                 'target':target_doc})
                nb_lines_written += 1
                
                if nb_lines_written % self.lines_per_file == 0 and nb_lines_written > 0:
                    f.close()
                    nb_lines_written = 0
                    nb_files_written += 1
                    out_filename_str = out_filename+'-'+str(nb_files_written).zfill(4)
                    f = open(out_filename_str,'w',encoding='utf8')
                    writer = csv.DictWriter(f, fieldnames=['input','target'])
                    writer.writeheader()
                    print('Processing',in_filename,'-->',out_filename_str)
            else:
                f.close()


def process_IS(d):
    def process_doc(doc):
            doc = re.sub(r"(\.|\!)([A-Z|-])", r"\1 \2", doc)
            doc = re.sub(r"\(KUVA:[^\)]+\)", "", doc)
            doc = re.sub(r"[\n\r\t]", " ", doc)
            doc = doc.strip()
            return(doc)
    try:
        input_doc = process_doc(d['content'])
        target_doc = process_doc(d['title'])
    except:
        input_doc = ''
        target_doc = ''
    finally:
        return(input_doc,target_doc)
    
def test():
    in_folder = './data/feed/raw'
    out_folder = './data/feed/processed2/IS'
    fp = FeedProcessor(in_folder,out_folder,process_IS)
    fp.process()
test()