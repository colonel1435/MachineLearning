#!usr/bin/env python3.4
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Administrator
# #  Description : 
# #  Time        : 2017/5/2

from functools import reduce
import numpy as np
from  numpy import random
from utils import DataLoader
from datetime import datetime

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node(object):
    def __init__(self, layer_index, node_index):
        '''
        Construct Node object
        layer_index: index of layer
        node_index: index of node
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        Set output, if node belongs to input layer
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        Add connect to downstraem
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        Add connect to upstream
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        output y = sigmoid(wx)
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        Hide layer, delta = Gradient *（∑（Ｗkiδk））
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        Output layer，delta = Gradient * δ
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        Print node info
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        Construct Const node
        layer_index: index of layer
        node_index: index of node
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        '''
        Add connect to downstream
        '''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        Hide layer, delta = Gradient *（∑（Ｗkiδk））
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        Print node info
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

class Layer(object):
    def __init__(self, layer_index, node_count):
        '''
        Init layer
        layer_index: index of layer
        node_count: number of node in layer
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        Set output, if it's input layer
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        Compute output vector of layer
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        Print layer info
        '''
        for node in self.nodes:
            print (node)

class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        Init connection, initial weight is a small random number
        upstream_node: nodes of upstream
        downstream_node: nodes of downstream
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node

        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        Compute gradient
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        '''
        Get current gradient
        '''
        return self.gradient

    def update_weight(self, rate):
        '''
        Update weight according to gradient decline algorithm
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        Print connect info
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print (conn)

class Network(object):
    def __init__(self, layers):
        '''
        Init full connected neural network
        layers: 2-d array，contains nodes of per layer
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0;
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)


    def train(self, labels, data_set, rate, iteration):
        '''
        Training neural network
        labels: training labels
        data_set: training charactor
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        Training network with one sample
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        '''
        Compute delta of per node
        '''
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        Update weight of per connection
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        Compute gradient of per connection
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        Get gradient
        label: sample label
        sample: sample input
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        '''
        Predict output
        sample: sample input vectors
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def dump(self):
        '''
        Print network info
        '''
        for layer in self.layers:
            layer.dump()
def gradient_check(network, sample_feature, sample_label):
    '''
    Check gradient
    network: neural net
    sample_feature: sample feature
    sample_label: sample label
    '''
    # Compute network delta
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b,
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2)))

    # Get gradient of per connection in current sample
    network.get_gradient(sample_feature, sample_label)

    # Check gradient of per weight
    for conn in network.connections.connections:
        # Get the gradient of connection
        actual_gradient = conn.get_gradient()

        # Compute delta of network though adding a small number
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # Compute delta of network though sub a small number
        conn.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)

        # Compute expected gradient
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # Print gradient
        print ('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))

# Full-connectted neural network
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator):
        '''
        Construct network
        input_size: dimen of input vector
        output_size: dimen of output vector
        activator: active function
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # weight array
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        # bias
        self.b = np.zeros((output_size, 1))
        # output vector
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        forward-compute
        input_array: input vector
        '''
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        backward-compute gradient of weight and bias
        delta_array: delta from previous layer
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        Update weight with gradient decline algorithm
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

# Sigmoid
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


# Neural network
# class Network(object):
#     def __init__(self, layers):
#         '''
#         构造函数
#         '''
#         self.layers = []
#         for i in range(len(layers) - 1):
#             self.layers.append(
#                 FullConnectedLayer(
#                     layers[i], layers[i+1],
#                     SigmoidActivator()
#                 )
#             )
#
#     def predict(self, sample):
#         '''
#         使用神经网络实现预测
#         sample: 输入样本
#         '''
#         output = sample
#         for layer in self.layers:
#             layer.forward(output)
#             output = layer.output
#         return output
#
#     def train(self, labels, data_set, rate, epoch):
#         '''
#         训练函数
#         labels: 样本标签
#         data_set: 输入样本
#         rate: 学习速率
#         epoch: 训练轮数
#         '''
#         for i in range(epoch):
#             for d in range(len(data_set)):
#                 self.train_one_sample(labels[d],
#                     data_set[d], rate)
#
#     def train_one_sample(self, label, sample, rate):
#         self.predict(sample)
#         self.calc_gradient(label)
#         self.update_weight(rate)
#
#     def calc_gradient(self, label):
#         delta = self.layers[-1].activator.backward(
#             self.layers[-1].output
#         ) * (label - self.layers[-1].output)
#         for layer in self.layers[::-1]:
#             layer.backward(delta)
#             delta = layer.delta
#         return delta
#
#     def update_weight(self, rate):
#         for layer in self.layers:
#             layer.update(rate)


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = DataLoader.get_training_data_set()
    test_data_set, test_labels = DataLoader.get_test_data_set()
    network = Network([784, 300, 10])
    print("start trainning")
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print ('%s epoch %d finished' % (datetime.now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print ('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_evaluate()
