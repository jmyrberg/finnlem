# -*- coding: utf8 -*-
"""Tensorflow implementation for sequence-to-sequence.

Modified from github.com/JayParks/tf-seq2seq.
"""


import os
import math

import numpy as np

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.util import nest

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from utils import create_folder, get_default_args


class Seq2SeqModel(object):

    def __init__(self, 
                 model_dir='./',
                 mode='train', 
                 cell_type='lstm', 
                 hidden_dim=256,
                 attn_dim=None,
                 embedding_dim=128, 
                 depth=2, 
                 attn_type='bahdanau', 
                 attn_input_feeding=True,
                 use_residual=False, 
                 reverse_source=True,
                 dropout_rate=0.2,
                 dtype='float32',
                 num_encoder_symbols=30000, 
                 num_decoder_symbols=30000,
                 start_token=2, 
                 end_token=1, 
                 pad_token=0, 
                 optimizer='adam', 
                 learning_rate=0.0001,
                 max_gradient_norm=1.0,
                 keep_every_n_hours=1,
                 beam_width=1, 
                 max_decode_step=30):

        # Train location
        self.model_dir = model_dir

        # Toggle between train and decode
        self.mode = mode.lower()

        # Network building
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.attn_dim = hidden_dim if attn_dim is None else attn_dim
        self.embedding_dim = embedding_dim
        self.depth = depth
        
        self.attn_type = attn_type
        self.attn_input_feeding = attn_input_feeding
        self.use_residual = use_residual
        self.reverse_source = reverse_source
        
        self.dropout_rate = dropout_rate
        
        self.dtype = tf.float16 if dtype=='float16' else tf.float32
        
        # Obtained from dictionary_
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        
        # Training
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.keep_every_n_hours = keep_every_n_hours
        
        # Decoding
        self.beam_width = beam_width
        self.use_beamsearch_decode = self.beam_width > 1 and self.mode == 'decode'
        self.max_decode_step = max_decode_step
        
        self._build_model()
       
    def _build_model(self):
        # Building encoder and decoder networks
        self._init_attributes()
        self._init_placeholders()
        self._build_encoder()
        self._build_decoder()
        self._init_session()

    def _init_attributes(self):
        self.use_dropout = True if self.dropout_rate == 0.0 else False
        self.keep_prob = 1.0 - self.dropout_rate
        self.n_params = np.sum([np.prod(v.get_shape().as_list()) 
                                for v in tf.trainable_variables()])

    def _init_placeholders(self):
        
        # Training progress
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(None, None), name='encoder_inputs')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')
        
        # Reverse encoder inputs
        if self.reverse_source:
            self.encoder_inputs = tf.reverse_sequence(self.encoder_inputs, 
                                    seq_lengths=self.encoder_inputs_length,
                                    seq_dim=1, batch_dim=0)

        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')
        
        # Dynamic batch size
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        
        # Train placeholders
        if self.mode == 'train':

            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None, None), name='decoder_inputs')
            
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

            # decoder_start_token: [batch_size,1]
            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.start_token          

            # Insert <SOS> symbol in front of each decoder input
            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                  self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # Create decoder targets
            max_time_steps = tf.shape(self.decoder_inputs)[1]
            # Add one timestep max_time_steps -> max_time_steps + 1
            end_padding = tf.ones(shape=[self.batch_size,1], dtype=tf.int32) * self.pad_token
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                    end_padding], axis=1)
            # decoder_end_token: [batch_size,max_time_steps + 1]
            decoder_end_token = tf.one_hot(
                indices=self.decoder_inputs_length, depth=max_time_steps+1,
                on_value=self.end_token, off_value=0, axis=-1, dtype=tf.int32)
            # Insert <EOS> symbol at the end of each decoder input
            # decoder_targets_train: [batch_size, max_time_steps + 1]
            self.decoder_targets_train = self.decoder_targets_train + decoder_end_token

    def _build_single_cell(self):
        if self.cell_type == 'lstm':
            cell_type = LSTMCell
        elif self.cell_type == 'gru':
            cell_type = GRUCell
        cell = cell_type(self.hidden_dim)
        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)
        return cell

    def _build_encoder_cell (self):
        return MultiRNNCell([self._build_single_cell() for _ in range(self.depth)])

    def _build_encoder(self):
        with tf.variable_scope('encoder'):
            # Encoder cell
            self.encoder_cell = self._build_encoder_cell()
            
            # Initialize encoder_embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
             
            # encoder_embeddings: [num_encoder_symbols, embedding_dim]
            self.encoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_encoder_symbols, self.embedding_dim],
                initializer=initializer, dtype=self.dtype)
            
            # encoder_inputs_embedded: [batch_size, time_steps, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings, ids=self.encoder_inputs)
       
            # Input projection layer to feed embedded inputs to the cell
            # Essential when use_residual=True to match input/output dims
            # input_layer: [batch_size, hidden_dim]
            input_layer = Dense(self.hidden_dim, dtype=self.dtype, name='input_projection')

            # Embedded inputs having gone through input projection layer
            # encoder_inputs_embedded: [batch_size, hidden_dim]
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)
    
            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_steps, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype,
                time_major=False, swap_memory=False)

    def _build_decoder_cell(self):
        
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length
        
        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        if self.attn_type == 'bahdanau':
            self.attention_mechanism = attention_wrapper.BahdanauAttention(
                num_units=self.attn_dim, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length,) 
            output_attention = False
        # 'Luong' style attention: https://arxiv.org/abs/1508.04025
        elif self.attn_type == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.attn_dim, memory=encoder_outputs, 
                memory_sequence_length=encoder_inputs_length,)
            output_attention = True
 
        # Building decoder_cell
        self.decoder_cell_list = [
            self._build_single_cell() for _ in range(self.depth)]

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            if self.use_residual:
                _input_layer = Dense(self.hidden_dim, dtype=self.dtype,
                                     name='attn_input_feeding')
                return _input_layer(array_ops.concat([inputs, attention], -1))
            else:
                return array_ops.concat([inputs, attention], -1)

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=None,
            cell_input_fn=attn_decoder_input_fn,
            output_attention=output_attention,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
                     else self.batch_size * self.beam_width
                         
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
          batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def _build_decoder(self):
        with tf.variable_scope('decoder'):
            # Decoder cell and initial state
            self.decoder_cell, self.decoder_initial_state = self._build_decoder_cell()

            # Initialize decoder embeddings to have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
             
            # decoder_embeddings: [num_decoder_symbols, embedding_dim]
            self.decoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_decoder_symbols, self.embedding_dim],
                initializer=initializer, dtype=self.dtype)

            # Input projection layer to feed embedded inputs to the cell
            # Essential when use_residual=True to match input/output dims
            # input_layer: [batch_size, hidden_dim]
            input_layer = Dense(self.hidden_dim, dtype=self.dtype, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            # output_layer: [batch_size, num_decoder_symbols]
            output_layer = Dense(self.num_decoder_symbols, name='output_projection')

            # Different decoder is needed in train / decode
            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_dim]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs_train)
               
                # Embedded inputs having gone through input projection layer
                # decoder_inputs_embedded: [batch_size, hidden_dim]
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                   sequence_length=self.decoder_inputs_length_train,
                                                   time_major=False,
                                                   name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)
                    
                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput / namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols]
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train, self.decoder_last_state_train, 
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length,
                    swap_memory=True))
                 
                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output) 
                
                # Use argmax to extract decoder symbols to emit
                # decoder_pred_train: [batch_size, max_time_steps + 1]
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                # Masking for valid and padded time steps
                # masks: [batch_size, max_time_step + 1]
                self.masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train, 
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                # loss: scalar
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train, 
                                                  targets=self.decoder_targets_train,
                                                  weights=self.masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)
                tf.summary.scalar('loss',self.loss)
                
                self._init_optimizer()

            elif self.mode == 'decode':
        
                # start_tokens: [batch_size, 1]
                start_tokens = tf.tile([self.start_token], [self.batch_size])

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))
                    
                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=self.end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                               embedding=embed_and_input_proj,
                                                               start_tokens=start_tokens,
                                                               end_token=self.end_token,
                                                               initial_state=self.decoder_initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=output_layer)
                # Greedy decoder
                # decoder_outputs_decode: BasicDecoderOutput instance / namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols]
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                
                # Beamsearch decoder
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance / namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width]
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance / namedtuple(scores, predicted_ids, parent_ids)
                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,    # error occurs
                    maximum_iterations=self.max_decode_step))

                if not self.use_beamsearch_decode:
                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # decoder_pred_decode: [batch_size, max_time_step, 1]
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width]
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

    def _init_optimizer(self):
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def _init_session(self):
        if not hasattr(self,'sess'):
            create_folder(self.model_dir)
            self.save_path = os.path.join(self.model_dir,'','model')
            train_summary_dir = os.path.join(self.model_dir,'train_summary/')
            valid_summary_dir = os.path.join(self.model_dir,'valid_summary/')
            create_folder(train_summary_dir)
            create_folder(valid_summary_dir)
            
            self.saver = tf.train.Saver(
                keep_checkpoint_every_n_hours=self.keep_every_n_hours)
            self.sm = tf.train.SessionManager()
            self.init_op = tf.global_variables_initializer()
            self.summary_op = tf.summary.merge_all()
            self.sess = self.sm.prepare_session("", init_op=self.init_op,
                            saver=self.saver, checkpoint_dir=self.model_dir)

            self.train_writer = tf.summary.FileWriter(train_summary_dir,self.sess.graph)
            self.valid_writer = tf.summary.FileWriter(valid_summary_dir,self.sess.graph)
    
    def _check_feeds(self, encoder_inputs, encoder_inputs_length, 
                    decoder_inputs, decoder_inputs_length, decode):
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                    "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                    "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}
    
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed  
    
    def get_params(self):
        return get_default_args(self.__init__)
    
    def train(self, encoder_inputs, encoder_inputs_length, 
              decoder_inputs, decoder_inputs_length):
        while True:# Session context

            if self.mode != 'train':
                raise ValueError("Train step can only be operated in train mode")
    
            input_feed = self._check_feeds(encoder_inputs, encoder_inputs_length,
                                          decoder_inputs, decoder_inputs_length, 
                                          decode=False)
            # Input feeds for dropout
            input_feed[self.keep_prob_placeholder.name] = self.keep_prob
            
            output_feed = [self.updates,
                           self.loss,
                           self.global_step,
                           self.summary_op]
            _,loss,global_step,summary_op = self.sess.run(output_feed, input_feed)
            
            self.train_writer.add_summary(summary_op, global_step)
            
            return loss,global_step

    def eval(self, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        input_feed = self._check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length,
                                      decode=False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
        
        output_feed = [self.loss,
                       self.global_step,
                       self.summary_op]
        
        loss,global_step,summary_op = self.sess.run(output_feed, input_feed)
        
        self.valid_writer.add_summary(summary_op, global_step)
        
        return loss,global_step

    def decode(self, encoder_inputs, encoder_inputs_length):
        input_feed = self._check_feeds(encoder_inputs, encoder_inputs_length, 
                                       decoder_inputs=None, decoder_inputs_length=None, 
                                       decode=True)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
 
        output_feed = [self.decoder_pred_decode]
        
        decoded_seqs, = self.sess.run(output_feed, input_feed)
        return decoded_seqs
    
    def save(self):
        self.saver.save(self.sess, self.save_path, 
                        global_step=self.global_step,
                        write_meta_graph=False)
        