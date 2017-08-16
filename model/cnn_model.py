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
CNN模型：只用Word数据，包括问题和描述
"""
class CNN():
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
            #self.cnn_input = tf.placeholder(tf.int32, [None, self.cnn_seq_length], name='cnn_input')
            self.rnn_input = tf.placeholder(tf.int32, [None, self.rnn_seq_length], name='rnn_input')
            self.labels = tf.placeholder(shape=(None, 1999), dtype=tf.float32, name="labels")
            self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")

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
        self.embedded_word_expanded = tf.expand_dims(self.embedded_word, -1)
        #[batch_size, rnn_seq_length, embed_size, 1]
        self.crnn_out = self.op_cnn_word()#[batch_size, self.num_filters_total]

        # Output layer
        with tf.variable_scope("output") as scope:
            self.output_W = self.get_weight_variable(
                shape=(self.num_filters_total, 1999), 
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
    
    def op_cnn_word(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("word_conv-pooling-%s" % filter_size):
                filter_ = self.get_weight_variable(name="word_filter-%s" % filter_size,
                                                   shape=[filter_size, self.embed_size, 1, self.num_filters])
                conv = tf.nn.conv2d(self.embedded_word_expanded, filter_, strides=[1,1,1,1],
                                    padding="VALID",name="conv") #shape:[batch_size,rnn_seq_length*2 - filter_size + 1,1,num_filters]
                b = tf.get_variable("word_b-%s"%filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv,b), "relu") #shape:[batch_size,rnn_seq_length*2 - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                pooled = tf.nn.max_pool(h, ksize=[1,self.rnn_seq_length-filter_size+1,1,1], 
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
