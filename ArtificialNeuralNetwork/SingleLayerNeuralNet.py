#!usr/bin/env python3.6.1
# -*- coding: utf-8 -*-
# #  FileName    :
# #  Author      : Mr.wumin
# #  Description :
# #  Time        : 2017/4/21
from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # 随机数发生器种子，以保证每次获得相同结果
        random.seed(1)

        # 对单个神经元建模，含有3个输入连接和一个输出连接
        # 对一个3 x 1的矩阵赋予随机权重值。范围-1～1，平均值为0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Sigmoid函数，S形曲线
    # 用这个函数对输入的加权总和做正规化，使其范围在0～1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Sigmoid函数的导数
    # Sigmoid曲线的梯度
    # 表示我们对当前权重的置信程度
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # 通过试错过程训练神经网络
    # 每次都调整突触权重
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # 将训练集导入神经网络
            output = self.think(training_set_inputs)

            # 计算误差（实际值与期望值之差）
            error = training_set_outputs - output

            # 将误差、输入和S曲线梯度相乘
            # 对于置信程度低的权重，调整程度也大
            # 为0的输入值不会影响权重
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # 调整权重
            self.synaptic_weights += adjustment

    # 神经网络一思考
    def think(self, inputs):
        # 把输入传递给神经网络
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # 初始化神经网络
    neural_network = NeuralNetwork()
    print ("随机的初始突触权重：{0}".format(neural_network.synaptic_weights))

    # 训练集。四个样本，每个有3个输入和1个输出
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # 用训练集训练神经网络
    # 重复一万次，每次做微小的调整
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("训练后的突触权重：{0}".format(neural_network.synaptic_weights))

    # 用新数据测试神经网络
    print ("考虑新的形势 [1, 0, 0] -> ?: {0}".format(neural_network.think(array([1, 0, 0]))))