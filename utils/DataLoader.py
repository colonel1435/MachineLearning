#!usr/bin/env python3.4
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Administrator
# #  Description : 
# #  Time        : 2017/5/2

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import struct
from bp import *
from datetime import datetime
import sys
import os

# Data loader
class Loader(object):
    def __init__(self, path, count):
        '''
        Init data loader
        path: file path
        count: number of file
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        Read file content
        '''
        print(self.path)
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        '''
        Change unsigned byte to int
        '''
        return struct.unpack('b', str(byte).encode("utf-8"))[0]


# Image data loader
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        Get image from file
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    content[start + i * 28 + j])
                    # self.to_int(content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        '''
        Change image to input sample vector
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        Load image data to get image vector
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set


# Label data loader
class LabelLoader(Loader):
    def load(self):
        '''
        Load label data to get label vector
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        Change label to 10-D vector
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    '''
    Get training data set
    '''
    image_loader = ImageLoader(os.path.join(os.path.pardir, "dataset\\train-images.idx3-ubyte"), 10)
    label_loader = LabelLoader(os.path.join(os.path.pardir, "dataset\\train-labels.idx1-ubyte"), 10)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    '''
    Get test data set
    '''
    image_loader = ImageLoader(os.path.join(os.path.pardir, "dataset\\t10k-images.idx3-ubyte"), 10)
    label_loader = LabelLoader(os.path.join(os.path.pardir, "dataset\\t10k-labels.idx1-ubyte"), 10)
    return image_loader.load(), label_loader.load()
