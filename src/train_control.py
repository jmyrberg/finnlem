# -*- coding: utf8 -*-
'''
Created on 14.7.2017

@author: Jesse
'''
import logging
import tensorflow as tf
from dictionary import Dictionary
from models import Seq2SeqModel
from utils import create_folder, list_files_in_folder
from os.path import exists,join
from model_train import get_model_config,get_model_config2
from data_utils import BatchGenerator

class TrainController():
    
    def __init__(self,
                 controller_name='TestModel',
                 base_dir='./training/',
                 dict_path='./data/dictionary.dict'):
        self.controller_name = controller_name
        self.base_dir = base_dir
        self.dict_path = dict_path
        
        self._init_controller()
        
    ########################## INITIALIZE ##########################
    def _init_controller(self):
        self.controller_dir = join(self.base_dir,self.controller_name)
        self.meta_path = join(self.controller_dir,self.controller_name)+'.meta'
        if not exists(self.controller_dir):
            self._init_new_controller()
        self._init_existing_controller()
        
    def _init_new_controller(self):
        # Paths and folders
        self.model_dir = join(self.controller_dir,'model')
        self.model_path = join(self.model_dir,self.controller_name)+'.model'
        self.to_train_files_path = join(self.controller_dir,'to_train.files')
        self.trained_files_path = join(self.controller_dir,'trained.files')
        self.log_path = join(self.controller_dir,self.controller_name)+'.log'
        
        # Create folders
        for folder in [self.controller_dir,self.model_dir]:
            create_folder(folder)
        
        # Log file
        self._init_logger()
        self.logger.info('Log file "%s" initialized',self.log_path)
        self._write_meta_file()
        
        # Create empty files
        for filename in [self.meta_path,
                         self.to_train_files_path,
                         self.trained_files_path]:
            try:
                open(filename,'x')
                self.logger.info('Created empty file %s',filename)
            except FileExistsError:
                self.logger.warn('Failed creating empty file %s,'+\
                                 'since it already exists',filename)
                pass
            
    def _init_existing_controller(self):
        self._read_meta_file()
        self._init_logger()
        self._read_train_files()
        self._load_dictionary()
            
    def _init_logger(self):
        logging.basicConfig(
            filename=self.log_path,
            filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG)
        self.logger = logging.getLogger('log')
        print('Logger initialized successfully')
        self.logger.info('Logger initialized at "%s"',self.log_path)    
    
    ########################## WRITE TO FILE ##########################
    def _write_meta_file(self):
        with open(self.meta_path,'w',encoding='utf8') as f:
            f.writelines('{} = {}\n'.format(k,v) 
                        for k,v in self._get_meta_dict().items())
        self.logger.info('Meta file "%s" updated',self.meta_path)
    
    def _write_train_files(self):
        self._write_to_train_file()
        self._write_trained_file()
        
    def _write_trained_file(self):
        with open(self.trained_files_path,'w',encoding='utf8') as f:
            f.writelines('{}\n'.format(fname) for fname in self.trained_files)
            
    def _write_to_train_file(self):
        with open(self.to_train_files_path,'w',encoding='utf8') as f:
            f.writelines('{}\n'.format(fname) for fname in self.to_train_files)

    ########################## SAVE DATA ##########################
    def _save_train_status(self):
        try:
            self._log_list_of_vars()
            start_global_step = self.global_step
            self.global_step += 1
            self.saver.save(self.sess, self.model_path, self.global_step,
                            write_meta_graph=False)
            self._write_trained_file()
            self.logger.info('Training status saved with global step %d',
                             self.global_step)
        except Exception as e:
            self.logger.error('Failed saving training status')
            self.logger.exception(e)
            if self.global_step > start_global_step:
                self.global_step -= 1
                self.logger.info('Global step decreased from %d to %d',
                                 start_global_step,self.global_step)
            raise e

    ########################## READ DATA ##########################
    def _read_meta_file(self):
        with open(self.meta_path,'r',encoding='utf8') as f:
            lines = f.read().splitlines()
        content = dict([tuple(line.split(" = ")) 
                        for line in lines if " = " in line])
        for k,v in content.items():
            setattr(self, k, v)
    
    def _read_train_files(self):
        with open(self.to_train_files_path,'r',encoding='utf8') as f:
            self.to_train_files = f.read().splitlines()
        with open(self.trained_files_path,'r',encoding='utf8') as f:
            self.trained_files = f.read().splitlines()
        self.logger.info('Training files "%s" and "%s" loaded successfully',
                         self.to_train_files_path,self.trained_files_path)
    
    ########################## LOAD DATA ##########################
    def _load_dictionary(self):
        try:
            d = Dictionary().load(self.dict_path)
            self.dictionary = d
            self.logger.info('Dictionary load successfully from "%s"',
                             self.dict_path)
        except:
            self.logger.log('Failed loading dictionary at "%s"',
                            self.dict_path)
            pass
    
    def _load_model_meta_graph(self):
        try:
            model_meta_graph_path = self.model_ckpt_path+'.meta'
            self.saver = tf.train.import_meta_graph(model_meta_graph_path)
            self.logger.info('Model meta graph loaded successfully from "%s"',
                             model_meta_graph_path)
        except:
            self.saver = tf.train.Saver()
            self.logger.warn('Could not load model meta graph from "%s"',
                             model_meta_graph_path)
    
    def _load_last_model_checkpoint(self):
        try:
            self.model_ckpt_path = tf.train.latest_checkpoint(self.model_dir+'/')
            self.global_step = int(self.model_ckpt_path.split('-')[-1])
            self.logger.info('Latest model checkpoint path "%s" found',
                            self.model_ckpt_path)
        except:
            self.model_ckpt_path = None
            self.global_step = 0
            self.logger.warn('Could not find existing model' + \
                             'checkpoint from "%s"',self.model_dir+'/')
            pass

    def _load_model_config(self,mode):
        try:
            self.model_config = get_model_config2(self.dictionary)
            self.model_config['mode'] = mode
            self.logger.info('Model config loaded successfully')
        except Exception as e:
            self.logger.error('Could not load model config')
            self.logger.exception(e)
            pass

    def _load_existing_model(self):
        try:
            self._load_last_model_checkpoint()
            self.saver.restore(self.sess, self.model_ckpt_path)
            self.logger.info('Existing model loaded successfully')
        except:
            self.logger.warn('Existing model could not be loaded')
            pass
        
    def _load_model(self,mode='train'):
        try:
            if hasattr(self,'model'):
                self.sess.close()
                del self.sess
                del self.model
                del self.model_config
                del self.saver
            self._load_model_config(mode=mode)
            self._create_seq2seq_model()
            self._create_training_session()
            self.sess.run(tf.global_variables_initializer())
            self._load_existing_model()
            self._log_list_of_vars()
        except Exception as e:
            self.logger.error('Could not load model')
            self.logger.exception(e)

    ########################## TRAINING HELPERS ##########################
    def _create_training_session(self):
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.logger.info('tf session and saver created successfully')
        
    def _create_seq2seq_model(self):
        try:
            self.model = Seq2SeqModel(config=self.model_config)
            self.logger.info('Seq2SeqModel created successfully')
        except Exception as e:
            self.logger.error('Could not create Seq2SeqModel')
            self.logger.exception(e)
            pass

    def _check_train_validity(self):
        # @TODO: check is everything is OK to start training
        return(True)

    ########################## GENERIC HELPERS ##########################
    def _get_meta_dict(self):
        return({
            'controller_dir':self.controller_dir,
            'model_dir':self.model_dir,
            'dict_path':self.dict_path,
            'model_path':self.model_path,
            'meta_path':self.meta_path,
            'to_train_files_path':self.to_train_files_path,
            'trained_files_path':self.trained_files_path,
            'log_path':self.log_path})
        
    def _log_list_of_vars(self):
        try:
            list_of_vars = [v.name for v in tf.global_variables()]
            self.logger.info('Number of tf vars (%d)',len(list_of_vars))
        except:
            self.logger.warn('Failed trying to print tf vars!')
            
    ################### USER-FACING FUNCTIONS ####################
    
    #### TRAINING ####
    def train(self, 
              batch_size=32, 
              max_seq_len=6000, 
              file_batch_size=1024*8, 
              save_every_n_batch=1000):
        if self._check_train_validity():
            self.logger.info('Starting to train...')
            self._load_model(mode='train')
            n_trained_files = len(self.trained_files)
            bg = BatchGenerator(self.dictionary).train_from_files(
                batch_size=batch_size,
                file_batch_size=file_batch_size,
                max_seq_len=max_seq_len,
                files=self.to_train_files,
                output_list=self.trained_files)
            
            for batch_nb,(input_batch,input_batch_lens,target_batch,target_batch_lens) in enumerate(bg):
#                 print(input_batch[-1])
#                 print(input_batch_lens[-1])
#                 print(target_batch[-1])
#                 print(target_batch_lens[-1])
                loss,_ = self.model.train(self.sess,input_batch,input_batch_lens,target_batch,target_batch_lens)
                n_trained_files_new = len(self.trained_files)
                print('%d files trained | Current batch %d (size %d) with loss of %f' % \
                      (n_trained_files_new,batch_nb,len(input_batch),loss))
                
                #print(self.model.eval(self.sess,input_batch,input_batch_lens,target_batch,target_batch_lens))
                
                if n_trained_files_new > n_trained_files:
                    self.logger.info('File "%s" trained successfully, model loss: %f',
                                     self.trained_files[-1],loss)
                    self._save_train_status()
                    n_trained_files = n_trained_files_new
                    
                if batch_nb % save_every_n_batch == 0:
                    self.logger.info('Batch number %d trained successfully, model loss: %f',
                                     batch_nb,loss)
                    self._save_train_status()

    def predict(self,docs,batch_size=32):
        self.logger.info('Starting to predict...')
        self._load_model(mode='decode')
        bg = BatchGenerator(self.dictionary).predict_from_docs(docs,
                                                               batch_size=batch_size)
        for input_batch,input_batch_lens in bg:
            pred = self.model.predict(self.sess, input_batch, input_batch_lens)
            pred = pred.max(2)
            doc = list(self.dictionary.seqs2docs(pred))[-1]
            clean_doc = "".join([e for e in doc if e not in ['<PAD>','<EOS>']])
            print('Prediction:',clean_doc)

    #### SETTERS ####
    def set_dict_path(self,dict_path):
        if exists(dict_path):
            self.dict_path = dict_path
            self._write_meta_file()
    
    #### ADD AND REMOVE ####
    def add_train_files(self,train_files,allow_copies=True):
        for file in train_files:
            if not allow_copies and file not in self.trained_files and exists(file):
                self.to_train_files.append(file)
            else:
                self.to_train_files.append(file)
        self._write_train_files()
                
    def remove_train_files(self,train_files):
        for file in train_files:
            if file in self.to_train_files:
                self.to_train_files.remove(file)
        self._write_train_files()
                
# tc = TrainController()
# tc.add_train_files(list_files_in_folder('./data/feed/processed2'),
#                    allow_same=False)
# tc.train()


