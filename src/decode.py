# -*- coding: utf8 -*-
"""Sequence to sequence model decoding"""


import argparse

from seq2seq import Seq2Seq
from utils import get_path_files
from data_utils import read_files_cycled, rebatch


parser = argparse.ArgumentParser('Model decoding')

# Path parameters
parser.add_argument("--model-dir", required=True,
                    type=str, action='store',
                    help='Model checkpoint and log save path')
parser.add_argument("--source-data-path", required=True,
                    type=str, action='store',
                    help='Path to source data to decode')
parser.add_argument("--decoded-data-path", required=True,
                    type=str, action='store',
                    help='Path to existing Dictionary')

# Path parameters (optional)
parser.add_argument("--dict-path", default=None,
                    type=str, action='store',
                    help='Path to existing Dictionary')

# Decoder parameters
parser.add_argument("--beam-width", default=1,
                    type=int, action='store',
                    help='Path to source data to decode')
parser.add_argument("--max-decode-step", default=30,
                    type=int, action='store',
                    help='Path to source data to decode')
parser.add_argument("--batch-size", default=32,
                    type=int, action='store',
                    help='Batch size to feed into model')
parser.add_argument("--file-batch-size", default=8192,
                    type=int, action='store',
                    help='Number of rows to read in-memory from each file')
parser.add_argument("--max-file-pool-size", default=50,
                    type=int, action='store',
                    help='Maximum number of files to cycle at a time')


def decode_model(args):

    # Model
    model = Seq2Seq(model_dir=args.model_dir)
    
    # Files to be decoded
    files = get_path_files(args.source_data_path)
    
    # Batch generator
    # File batches
    file_gen = read_files_cycled(
        filenames=files,
        max_file_pool_size=args.max_file_pool_size,
        file_batch_size=args.file_batch_size, 
        file_batch_shuffle=args.shuffle_file_batches)
    
    # Decode batches
    decode_gen = rebatch(
        file_gen, 
        in_batch_size_limit=args.file_batch_size*args.max_file_pool_size,
        out_batch_size=args.batch_size, 
        shuffle=args.shuffle_file_batches,
        flatten=True)

    # Decode
    write_target = False
    for batch_nb,batch in enumerate(decode_gen):
        
        if batch_nb == 0:
            
            # Number of columns in batch
            n_cols = len(batch[0])
            if n_cols == 1:
                source_docs = batch
            elif n_cols == 2:
                source_docs,target_docs = batch
                write_target = True
            else:
                raise ValueError("Number of columns found %d not in [1,2]" \
                                  % n_cols)
                
            # Output file handle and headers
            fout = open(args.decoded_data_path, 'w', encoding='utf8')
            fout.write('source\t')
            if write_target:
                fout.write('target\t')
            fout.write("\t".join([str(k) for k in args.beam_width])+'\n')
            
        # Get decoded documents: list of lists, with beams as elements
        decoded_docs = model.decode(source_docs)
        
        # Write beams to file
        for i in len(decoded_docs):
            fout.write(source_docs[i]+'\t')
            if write_target:
                fout.write(target_docs[i]+'\t')
            for k in args.beam_width:
                decoded_doc = decoded_docs[i][k]
                fout.write(decoded_doc+'\t')
            fout.write('\n')
            
            
def main():
    args = parser.parse_args()
    decode_model(args)
    
if __name__=='__main__':
    main()