# -*- coding: utf8 -*-
'''
Created on 14.7.2017

@author: Jesse
'''
import json
import logging
import tensorflow as tf

from os.path import exists, join

from dictionary.dictionary import Dictionary
from models.seq2seq import Seq2Seq
from utils import create_folder, update_dict


class TrainController():
    
    def __init__(self,
                 controller_name='TestModel',
                 base_dir='./training/',
                 dict_path='./data/dictionary2.dict',
                 model_config={}):
        self.controller_name = controller_name
        self.base_dir = base_dir
        self.dict_path = dict_path
        self.model_config = model_config
        
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
        self.model_config_path = join(self.model_dir,self.controller_name)+'.config'
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
                         self.model_config_path,
                         self.to_train_files_path,
                         self.trained_files_path]:
            try:
                open(filename,'x')
                self.logger.info('Created empty file %s',filename)
            except FileExistsError:
                self.logger.warn('Failed creating empty file %s,'+\
                                 'since it already exists',filename)
                pass
            
        # Write given model config to file
        self._write_model_config_file()
            
    def _init_existing_controller(self):
        self._read_meta_file()
        self._init_logger()
        self._read_train_files()
        self._read_model_config()
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

    def _write_model_config_file(self):
        try:
            print(self.model_config)
            with open(self.model_config_path,'w') as f:
                f.write(json.dumps(self.model_config, indent=4))
            self.logger.info('Model config file "%s" written successfully',
                             self.model_config_path)
        except:
            self.logger.warn('Could not write model config to file "%s"',
                             self.model_config_path)
    
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
        
    def _read_model_config(self):
        try:
            with open(self.model_config_path,'r') as f:
                self.model_config = json.loads(f.read())
            self.logger.info('Model config loaded successfully')
        except Exception as e:
            self.logger.error('Could not load model config')
            self.logger.exception(e)
            pass
    
    ########################## LOAD DATA ##########################
    def _load_dictionary(self):
        try:
            d = Dictionary().load(self.dict_path)
            self.dictionary = d
            self.logger.info('Dictionary load successfully from "%s"',
                             self.dict_path)
        except:
            self.logger.log('Failed loading dictionary2 at "%s"',
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

    def _load_existing_model(self):
        try:
            self._load_last_model_checkpoint()
            self.saver.restore(self.sess, self.model_ckpt_path)
            self.logger.info('Existing model loaded successfully')
        except:
            self.logger.warn('Existing model could not be loaded')
            pass
        
    def _load_model(self,mode,params):
        try:
            if hasattr(self,'model'):
                self.sess.close()
                del self.sess
                del self.model
                del self.model_config
                del self.saver
            self._read_model_config()
            self._temp_update_model_config(mode,params)
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
            self.model = Seq2Seq(**self.model_config)
            self.logger.info('Seq2SeqModel created successfully')
        except Exception as e:
            self.logger.error('Could not create Seq2SeqModel')
            self.logger.exception(e)
            pass

    def _check_train_validity(self):
        # @TODO: check is everything is OK to start training
        return(True)

    ########################## GENERIC HELPERS ##########################
    def _add_dictionary_to_model_config(self):
        self.model_config['num_encoder_symbols'] = self.dictionary.n_tokens
        self.model_config['num_decoder_symbols'] = self.dictionary.n_tokens
        self.model_config['start_token'] = self.dictionary.EOS
        self.model_config['end_token'] = self.dictionary.EOS
        self.model_config['pad_token'] = self.dictionary.PAD
        
    def _temp_update_model_config(self,mode,params):
        try:
            self._add_dictionary_to_model_config()
            # @ TODO: Check that we dont overwrite originals
            params['mode'] = mode
            self.model_config = update_dict(self.model_config, params)
            self.logger.info('Model config temporarily updated:\n%s',
                             self.model_config)
        except:
            self.logger.error('Model config could not be updated')
    
    def _get_meta_dict(self):
        return({
            'controller_dir':self.controller_dir,
            'model_dir':self.model_dir,
            'dict_path':self.dict_path,
            'model_path':self.model_path,
            'meta_path':self.meta_path,
            'model_config_path':self.model_config_path,
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
    def train(self, train_generator, 
              save_every_n_batch=100,
              opt_params={}):
        if self._check_train_validity():
            self.logger.info('Starting to train...')
            
            self._load_model(mode='train',params=opt_params)
    
            n_trained_files = len(self.trained_files)
            for batch_nb,(input_batch,input_batch_lens,
                          target_batch,target_batch_lens) in enumerate(train_generator):
                loss = self.model.train(self.sess,
                                          input_batch,input_batch_lens,
                                          target_batch,target_batch_lens)
                n_trained_files_new = len(self.trained_files)
                print('%d files trained | Current batch %d (size %d) with loss of %f' % \
                      (n_trained_files_new,batch_nb,len(input_batch),loss))
                
                if n_trained_files_new > n_trained_files:
                    self.logger.info('File "%s" trained successfully, model loss: %f',
                                     self.trained_files[-1],loss)
                    self._save_train_status()
                    n_trained_files = n_trained_files_new
                    
                if batch_nb % save_every_n_batch == 0:
                    self.logger.info('Batch number %d trained successfully, model loss: %f',
                                     batch_nb,loss)
                    self._save_train_status()

    def decode(self, decode_generator, batch_size=32,
               decode_params={}):
        self.logger.info('Starting to predict...')
        
        self._load_model(mode='decode',params=decode_params)
        
        for (doc_ar,source_seqs,source_lens,target_seqs,target_lens) in decode_generator:
            pred = self.model.predict(self.sess, source_seqs, source_lens)
            for (source,target),pred_seq in zip(doc_ar,pred):
                print('\nSource:',source)
                print('Target:',target)
                for k in range(pred_seq.shape[1]):
                    pred_doc = self.dictionary.seq2doc(pred_seq[:,k])
                    print('Pred %d: %s' % (k+1,pred_doc))
                

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


