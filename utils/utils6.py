#-*- coding:utf-8 -*-
import numpy as np
import os
import pandas as pd
import math
from six.moves import xrange

class Utils():
    def __init__(self, words_num, chars_num, batch_size, eval_batch_size, 
                 dataset_dir, epoch_num, word_dict, char_dict, shuffle=True):
        self.words_num = words_num
        self.chars_num = chars_num
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.epoch_num = epoch_num
        self.word_dict = word_dict
        self.char_dict = char_dict
        
        self.init_all_reader()
        
    def init_all_reader(self):
        self.all_reader = pd.DataFrame(columns=[0,1,2,3,4])
        for i in range(9):
            train_set_path = self.dataset_dir + 'all_info_block_%d.txt' % i
            reader = pd.read_table(train_set_path, sep='\t', header=None)
            self.all_reader = pd.concat([self.all_reader,reader], axis=0)
        self.all_reader.fillna('', inplace=True)
    # 生成训练数据，所有数据
    def generate_t_batch(self):
        for epoch in range(self.epoch_num):
            if self.shuffle:
                t_all_reader = self.all_reader.sample(frac=1)
            num_batches_per_epoch = int((t_all_reader.shape[0] - 1) / self.batch_size) + 1
            #print(t_all_reader.shape[0])
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.batch_size
                end_index = min((batch_num + 1) * self.batch_size, t_all_reader.shape[0])
                reader = t_all_reader.iloc[start_index:end_index, :]
                y = []
                x_char1 = []
                x_char2 = []
                x_word_tit = []
                x_word_len_tit = []
                x_word_des = []
                x_word_len_des = []
                for i in xrange(reader.shape[0]):
                    # 按','切分标签
                    temp = reader.iloc[i][4].split(',')                
                    # 如果分类数大于5，只取前5个分类
                    if (len(temp)>5):
                        temp = temp[0:5]
                    # 设置标签的对应位置为1，其余位置为0
                    label = np.zeros(1999)
                    for temp_label in temp:
                        label[int(temp_label)] = 1
                    y.append(label)
                    
                    temp1 = reader.iloc[i][0].split(',')
                    a = self.char_padding_sequence(temp1) 
                    x_char1.append(a)
                    temp2 = reader.iloc[i][2].split(',')
                    b = self.char_padding_sequence(temp2)
                    x_char2.append(b)
                    
                    temp3 = reader.iloc[i][1].split(',')
                    c, d = self.word_padding_sequence(temp3)
                    x_word_tit.append(c)
                    x_word_len_tit.append(d)
                    
                    temp4 = reader.iloc[i][3].split(',')
                    e, f = self.word_padding_sequence(temp4)
                    x_word_des.append(e)
                    x_word_len_des.append(f)
                data = list(zip(x_char1, x_char2, x_word_tit, x_word_des, x_word_len_tit, x_word_len_des, y))
                data = np.array(data)
                yield data
                
    # 生成批次数据
    def generate_e_batch(self):
        # 读取训练数据和标签
        all_reader = pd.read_table(self.dataset_dir + 'all_info_block_9.txt', sep='\t', header=None)
        all_reader.fillna('', inplace=True)
        while 1:
            y = []
            x_char1 = []
            x_char2 = []
            x_word_tit = []
            x_word_len_tit = []
            x_word_des = []
            x_word_len_des = []
            real_labels = [] # 用于evaluation计算
            reader = all_reader.sample(n=self.eval_batch_size)
            for i in xrange(reader.shape[0]):
                # 按','切分标签
                temp = reader.iloc[i][4].split(',')
                # 如果分类数大于5，只取前5个分类
                if (len(temp)>5):
                    temp = temp[0:5]
                temp = map(int, temp)
                real_labels.append(temp)
                # 设置标签的对应位置为1，其余位置为0
                label = np.zeros(1999)
                for temp_label in temp:
                    label[int(temp_label)] = 1
                y.append(label)
                
                temp1 = reader.iloc[i][0].split(',')
                a = self.char_padding_sequence(temp1) 
                x_char1.append(a)
                temp2 = reader.iloc[i][2].split(',')
                b = self.char_padding_sequence(temp2)
                x_char2.append(b)
                
                temp3 = reader.iloc[i][1].split(',')
                c, d = self.word_padding_sequence(temp3)
                x_word_tit.append(c)
                x_word_len_tit.append(d)
                
                temp4 = reader.iloc[i][3].split(',')
                e, f = self.word_padding_sequence(temp4)
                x_word_des.append(e)
                x_word_len_des.append(f)
                
            data = list(zip(x_char1, x_char2, x_word_tit, x_word_des, x_word_len_tit, x_word_len_des, y, real_labels))    
            data = np.array(data)
            yield data
            
    # 生成预测数据
    def generate_p_batch(self):
        reader = pd.read_table(self.dataset_dir + 'question_eval_set.txt', sep='\t', header=None)
        reader.fillna('', inplace=True)
        # 每个epoch的num_batch
        predict_set_num_batches = int((reader.shape[0] - 1) / self.eval_batch_size) + 1
        print("predict_set_num_batches:",predict_set_num_batches)
        for batch_num in range(predict_set_num_batches):
            start_index = batch_num * self.eval_batch_size
            end_index = min((batch_num + 1) * self.eval_batch_size, reader.shape[0])
            temp_reader = reader.iloc[start_index:end_index,:]
            # 生成测试数据
            x_char1 = []
            x_char2 = []
            x_word_tit = []
            x_word_len_tit = []
            x_word_des = []
            x_word_len_des = []
            # 读取训练数据和标签
            for i in xrange(temp_reader.shape[0]):
                temp1 = temp_reader.iloc[i][1].split(',')
                a = self.char_padding_sequence(temp1) 
                x_char1.append(a)
                temp2 = temp_reader.iloc[i][3].split(',')
                b = self.char_padding_sequence(temp2)
                x_char2.append(b)
                
                temp3 = temp_reader.iloc[i][2].split(',')
                c, d = self.word_padding_sequence(temp3)
                x_word_tit.append(c)
                x_word_len_tit.append(d)
                
                temp4 = temp_reader.iloc[i][4].split(',')
                e, f = self.word_padding_sequence(temp4)
                x_word_des.append(e)
                x_word_len_des.append(f)
                
            data = list(zip(x_char1, x_char2, x_word_tit, x_word_des, x_word_len_tit, x_word_len_des))    
            data = np.array(data)
            yield data
            
    # 填充
    def word_padding_sequence(self, xlist):
        try:
            xlist.remove('')
        except:
            pass
        rtn_list = []
        for x in xlist:
            try:
                rtn_list.append(self.word_dict[x])
            except:
                rtn_list.append(0)
        length = len(rtn_list)
        if length >= self.words_num:
            return rtn_list[0:self.words_num], self.words_num
        else:
            sup = self.words_num - length
            rtn_list = rtn_list + ([0]*sup)
            return rtn_list, length
    # 填充
    def char_padding_sequence(self, xlist):
        try:
            xlist.remove('')
        except:
            pass
        rtn_list = []
        for x in xlist:
            try:
                rtn_list.append(self.char_dict[x])
            except:
                rtn_list.append(0)
        length = len(rtn_list)
        if length >= self.chars_num:
            return rtn_list[0:self.chars_num]
        else:
            sup = self.chars_num - length
            rtn_list = rtn_list + ([0]*sup)
            return rtn_list
   
    # 知乎提供的评测方案
    def eval(self, predict_label_and_marked_label_list):
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
        #print ("Precision:", precision)
        try:
            rtn = (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            rtn = 0
        return rtn