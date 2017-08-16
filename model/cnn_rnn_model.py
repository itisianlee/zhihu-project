#-*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
import os
import pandas as pd
import pickle
import math
import time
"""
CNN-RNN模型：用CNN训练char数据，用rnn训练Word数据
"""
class CRNN_char_word():
    def __init__(self, cnn_seq_length, rnn_seq_length, batch_size, eval_batch_size, hidden_size, 
                 lr, reg_rate,epoch_num,save_per_step, eval_per_step, keep_prob, 
                 atn_hidden_size, shuffle, ckpt_path, num_sentences, 
                 filter_sizes, num_filters, decay_steps, decay_rate, vocab_size, char_size, embed_size, is_train):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.epoch_num = epoch_num
        self.reg_rate = reg_rate
        self.cnn_seq_length = cnn_seq_length
        self.rnn_seq_length = rnn_seq_length
        self.save_per_step = save_per_step
        self.eval_per_step = eval_per_step
        self.ckpt_path = ckpt_path
        self.output_keep_prob = keep_prob
        self.atn_hidden_size = atn_hidden_size
        self.shuffle = shuffle
        self.num_sentences = num_sentences
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.embed_size = embed_size
        self.is_train = is_train
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.
        self.initialize_placeholder()
        self.initialize_weights()
        self.inference()
        
    def initialize_placeholder(self):
        self.global_step = tf.Variable(0, name="global_step")
        with tf.variable_scope('placeholder') as scope:
            self.cnn_input = tf.placeholder(tf.int32, [None, self.cnn_seq_length], name='cnn_input')
            self.rnn_input = tf.placeholder(tf.int32, [None, self.rnn_seq_length*2], name='rnn_input')
            self.labels = tf.placeholder(shape=(None, 1999), dtype=tf.float32, name="labels")
            self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
            self.batch_length = tf.placeholder(shape=(1,), dtype=tf.int32, name="batch_length")  
            self.tit_input_len = tf.placeholder(shape=(None,), dtype=tf.int32, name="tit_input_length")
            self.des_input_len = tf.placeholder(shape=(None,), dtype=tf.int32, name="des_input_length")

    def initialize_weights(self):
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.word_Embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], 
                                                           -1.0, 1.0), name="word_Embedding_weights")
            self.char_Embedding = tf.Variable(tf.random_uniform([self.char_size, self.embed_size], 
                                                           -1.0, 1.0), name="char_Embedding_weights")

    def inference(self):
        # RNN----------------------RNN
        self.embedded_word = tf.nn.embedding_lookup(self.word_Embedding, self.rnn_input)
        #[batch_size, rnn_seq_length, embed_size]
        (self.embedded_tit,self.embedded_des) = tf.split(self.embedded_word, 2, axis=1)
        #[batch_size, rnn_seq_length/2, embed_size]
        self.rnn_tit_out = self.stacked_bidirectional_rnn(self.embedded_tit, 1, self.tit_input_len)
        self.rnn_des_out = self.stacked_bidirectional_rnn(self.embedded_des, 2, self.des_input_len)
        self.rnn_out = tf.concat((self.rnn_tit_out, self.rnn_des_out), axis=1)#[batch_size, 4*hidden_size]

        # CNN----------------------CNN
        self.embedded_char = tf.nn.embedding_lookup(self.char_Embedding, self.cnn_input)
        #[batch_size, cnn_seq_length, embed_size]
        self.embedded_char_expanded = tf.expand_dims(self.embedded_char, -1)
        #[batch_size, cnn_seq_length, embed_size, 1]
        self.cnn_out = self.op_cnn()

        self.crnn_out = tf.concat((self.rnn_out, self.cnn_out), 1)
        # Output layer
        with tf.variable_scope("output") as scope:
            self.output_W = self.get_weight_variable(
                shape=(self.hidden_size * 4 + self.num_filters_total, 1999), 
                name="output_w", 
                initializer=tf.contrib.layers.xavier_initializer())
            self.output_b = tf.Variable(tf.zeros(1999), name="output_b")
            self.logits = tf.nn.xw_plus_b(self.crnn_out, self.output_W, self.output_b, name="logits")
        # 定义loss
        with tf.name_scope("loss") as scope:
            _loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, 
                                                                           labels=self.labels))
            self.loss = _loss
            tf.summary.scalar('loss_train', self.loss)
        # 定义优化器
        with tf.name_scope("optimizer"):
            learning_rate = tf.train.exponential_decay(self.lr, self.global_step, 2000, 0.92, staircase=True)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, 
                                                                                   global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=24)
        self.merged = tf.summary.merge_all()

    def stacked_bidirectional_rnn(self, _inputs, flag, input_length):
        with tf.variable_scope("%d-GRU-ATN" % flag):
            encoder_fw = tf.contrib.rnn.GRUCell(self.hidden_size)
            encoder_bw = tf.contrib.rnn.GRUCell(self.hidden_size)
            initial_state_fw = encoder_fw.zero_state(self.batch_length, dtype=tf.float32)
            initial_state_bw = encoder_bw.zero_state(self.batch_length, dtype=tf.float32)
            encoder_fw = tf.contrib.rnn.DropoutWrapper(encoder_fw, output_keep_prob=self.output_keep_prob)
            encoder_bw = tf.contrib.rnn.DropoutWrapper(encoder_bw, output_keep_prob=self.output_keep_prob) 
            ((self.fw_output, self.bw_output), (self.fw_state, self.bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(encoder_fw, encoder_bw, _inputs, 
                                             input_length,
                                             initial_state_fw, 
                                             initial_state_bw, dtype=tf.float32))
            output = tf.concat((self.fw_output, self.bw_output), 2)#[batch_size, rnn_seq_length/2, 2*hidden_size]
            # attention layer
            _atn_in = tf.expand_dims(output, axis=2) 
            atn_W = self.get_weight_variable(shape=[1, 1, 2*self.hidden_size, self.atn_hidden_size], name="atn_W")
            #tf.summary.histogram('atn_W', atn_W)
            #tf.summary.scalar('atn_w', tf.reduce_mean(atn_W))
            atn_b = tf.Variable(tf.zeros(shape=[self.atn_hidden_size]))
            #tf.summary.histogram('atn_b', atn_b)
            #tf.summary.scalar('atn_b', tf.reduce_mean(atn_b))
            atn_v = self.get_weight_variable(shape=[1, 1, self.atn_hidden_size, 1], name="atn_v")
            atn_activations = tf.nn.tanh(
                tf.nn.conv2d(_atn_in, atn_W, strides=[1,1,1,1], padding='SAME') + atn_b)
            atn_scores = tf.nn.conv2d(atn_activations, atn_v, strides=[1,1,1,1], padding='SAME')
            atn_probs = tf.nn.softmax(tf.squeeze(atn_scores, [2, 3]))
            _atn_out = tf.matmul(tf.expand_dims(atn_probs, 1), output)#[batch_size, 1, 2*hidden_size]
            atn_out = tf.squeeze(_atn_out, [1], name="atn_out")
        return atn_out

    def op_cnn(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-pooling-%s" %filter_size):
                filter_ = self.get_weight_variable(name="filter-%s" % filter_size,
                                                   shape=[filter_size, self.embed_size, 1, self.num_filters])
                conv = tf.nn.conv2d(self.embedded_char_expanded, filter_, strides=[1,1,1,1],
                                    padding="VALID",name="conv") #shape:[batch_size,cnn_seq_length - filter_size + 1,1,num_filters]
                b = tf.get_variable("b-%s"%filter_size, [self.num_filters]) 
                h = tf.nn.relu(tf.nn.bias_add(conv,b), "relu") #shape:[batch_size,cnn_seq_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                pooled = tf.nn.max_pool(h, ksize=[1,self.cnn_seq_length-filter_size+1,1,1], 
                                        strides=[1,1,1,1], padding='VALID', name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 3) #shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat

    def get_weight_variable(self, shape, name, regularizer=None, 
                            initializer=tf.truncated_normal_initializer(stddev=0.1)):
        weights = tf.get_variable(shape=shape, name=name, initializer=initializer)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights
