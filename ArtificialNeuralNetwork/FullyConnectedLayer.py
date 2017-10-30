#!usr/bin/env python3.6
# -*- coding: utf-8 -*-
# #  FileName    :
# #  Author      : Administrator
# #  Description : Fully-connected neural network layer, prefer https://www.zybuluo.com/hanbingtao/note/476663
# #  Time        : 2017/10/29

import numpy as np

class FullyConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        '''
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # Weight
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # Bias

        self.b = np.zeros((output_size, 1))
        self.input = np.zeros((input_size, 1))
        self.output = np.zeros((output_size, 1))
        self.delta = mp.zeros((output_size, 1))

        def forward(self, input_array):
            self.input = input_array
            self.output = self.activator.forward(
                np.dot(self.W. input_array) + self.b
            )

        def backward(self, delta_array):
            self.delta = self.activator.backward(self.input_array *
                                                 np.dot(self.W.T, delta_array))
            self.W_grad = np.dot(delta_array, self.input.T)
            self.b_grad = delta_array

        def update(self, learning_rate):
            self.W += learning_rate * self.W_grad
            self.b += learning_rate * self.b_grad

class SigmoidActivitor():
    ''''''
    def forward(self, input):
        return 1.0 / (1.0 + np.exp(- input))

    def backward(self, output):
        return output * (1 - output)

class NetWork():
    '''
    
    '''
    def __init__(self, layers):
        self.layers = [
            FullyConnectedLayer(layers[i], layers[i + 1], SigmoidActivitor()) for i in range(len(layers) - 1)
        ]

    def predit(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.cal_gradient(label)
        self.update_weight(rate)

    def cal_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
