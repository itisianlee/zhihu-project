#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from config import Config
# from utils import *
import sys
import os
import pandas as pd
import pickle
import math
import time

# 生成预测数据
def generate_p_batch(config):
    reader = pd.read_table(config.dataset_dir + 'question_eval_set.txt', sep='\t', header=None)
    x_tit_text = reader.iloc[:,2]
    x_des_text = reader.iloc[:,4]
    # 每个epoch的num_batch
    predict_set_num_batches = int((reader.shape[0] - 1) / config.eval_batch_size) + 1
    print("predict_set_num_batches:",predict_set_num_batches)
    for batch_num in range(predict_set_num_batches):
        start_index = batch_num * config.eval_batch_size
        end_index = min((batch_num + 1) * config.eval_batch_size, reader.shape[0])
        # 生成测试数据
        pred_tit_x = []
        pred_des_x = []
        # 读取训练数据和标签
        sequence_tit_length = []
        sequence_des_length = []
        for line1,line2 in zip(x_tit_text[start_index:end_index], x_des_text[start_index:end_index]):
            try:
                temp1 = line1.split(',')
                a, b = padding_sequence(config.words_num, temp1)
                pred_tit_x.append(a)
                sequence_tit_length.append(b)
            except:
                pred_tit_x.append([0]*config.words_num)
                sequence_tit_length.append(0)
            try:
                temp2 = line2.split(',')
                c, d = padding_sequence(config.des_words_num, temp2)
                pred_des_x.append(c)
                sequence_des_length.append(d)
            except:
                pred_des_x.append([0]*config.words_num)
                sequence_des_length.append(0)
        pred_des_x = np.array(pred_des_x)
        pred_tit_x = np.array(pred_tit_x)
        data = list(zip(pred_tit_x, pred_des_x, sequence_tit_length, sequence_des_length))
        data = np.array(data)
        yield data
        
# 生成训练数据，所有数据
def generate_t_batch(config):
    all_reader = pd.DataFrame(columns=[0,1,2])
    for i in range(9):
        train_set_path = config.dataset_dir + 'title_des_topic_block_%d.txt' % i
        reader = pd.read_table(train_set_path, sep='\t', header=None)
        all_reader = pd.concat([all_reader,reader], axis=0)
    for epoch in range(config.epoch_num):
        if config.shuffle:
            all_reader = all_reader.sample(frac=1)
        num_batches_per_epoch = int(2699999 / config.batch_size) + 1
        sequence_tit_length = []
        sequence_des_length = []
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * config.batch_size
            end_index = min((batch_num + 1) * config.batch_size, 2700000)
            reader = all_reader.iloc[start_index:end_index, :]
            y = []
            x_tit_text = []
            x_des_text = []
            for i in xrange(reader.shape[0]):
                # 按','切分标签
                temp = reader.iloc[i][2].split(',')                
                # 如果分类数大于5，只取前5个分类
                if (len(temp)>5):
                    temp = temp[0:5]
                # 设置标签的对应位置为1，其余位置为0
                label = np.zeros(1999)
                for temp_label in temp:
                    label[int(temp_label)] = 1
                y.append(label)
                try:
                    temp1 = reader.iloc[i][0].split(',')
                    a, b = padding_sequence(config.words_num, temp1)
                    x_tit_text.append(a)
                    sequence_tit_length.append(b)
                except:
                    x_tit_text.append([0]*config.words_num)
                    sequence_tit_length.append(0)
                try:
                    temp2 = reader.iloc[i][1].split(',')
                    c, d = padding_sequence(config.des_words_num, temp2)
                    x_des_text.append(c)
                    sequence_des_length.append(d)
                except:
                    x_des_text.append([0]*config.words_num)
                    sequence_des_length.append(0)
            data = list(zip(x_tit_text, x_des_text, sequence_tit_length, sequence_des_length, y))
            data = np.array(data)
            yield data

# 生成批次数据
def generate_e_batch(config):
    # 读取训练数据和标签
    all_reader = pd.read_table(config.dataset_dir + 'title_des_topic_block_9.txt', sep='\t', header=None)
    while 1:
        eval_y = []
        x_tit_text = []
        x_des_text = []
        real_labels = [] # 用于evaluation计算
        sequence_tit_length = []
        sequence_des_length = []
        reader = all_reader.sample(n=config.eval_batch_size)
        for i in xrange(reader.shape[0]):
            # 按','切分标签
            temp = reader.iloc[i][2].split(',')
            # 如果分类数大于5，只取前5个分类
            if (len(temp)>5):
                temp = temp[0:5]
            temp = map(int, temp)
            real_labels.append(temp)
            # 设置标签的对应位置为1，其余位置为0
            label = np.zeros(1999)
            for temp_label in temp:
                label[int(temp_label)] = 1
            eval_y.append(label)
            try:
                temp1 = reader.iloc[i][0].split(',')
                a, b = padding_sequence(config.words_num, temp1)
                x_tit_text.append(a)
                sequence_tit_length.append(b)
            except:
                x_tit_text.append([0]*config.words_num)
                sequence_tit_length.append(0)
            try:
                temp2 = reader.iloc[i][1].split(',')
                c, d = padding_sequence(config.des_words_num, temp2)
                x_des_text.append(c)
                sequence_des_length.append(d)
            except:
                x_des_text.append([0]*config.words_num)
                sequence_des_length.append(0)
        data = list(zip(x_tit_text, x_des_text, sequence_tit_length, sequence_des_length, eval_y, real_labels))    
        data = np.array(data)
        yield data
        
# 知乎提供的评测方案
def eval(predict_label_and_marked_label_list):
    right_label_num = 0  #总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  #在各个位置上总命中数量
    sample_num = 0   #总问题数量
    all_marked_label_num = 0    #总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:     #命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1
    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        # 下标0-4 映射到 pos1-5 + 1，所以最终+2
        try:
            precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
        except ZeroDivisionError:
            precision += 0.0
    recall = float(right_label_num) / all_marked_label_num
    print ("Precision:", precision)
    try:
        rtn = (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        rtn = 0
    return rtn

# 填充
def padding_sequence(words_num, xlist):
    length = len(xlist)
    if length >= words_num:
        return xlist[0:words_num], words_num
    else:
        sup = words_num - length
        xlist = xlist + ([0]*sup)
        return xlist, length