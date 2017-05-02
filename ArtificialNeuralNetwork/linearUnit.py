#!usr/bin/env python3.6.1
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Mr.wumin
# #  Description : Refer to https://www.zybuluo.com/hanbingtao/note/448086
# #  Time        : 2017/4/21

from MachineLearning.ArtificialNeuralNetwork.Perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np
import decimal

class LinearUnit(Perceptron):
    def __init__(self, input_num, activator):
        '''Init linear unit'''
        Perceptron.__init__(self, input_num, activator)
		

def active_func(x):
    return x

def get_training_dataset():
    '''
    Build training dataset, like z = 1000x + 2000
    '''
    # Input vector which contains one param indacated work experience
    input_vecs = [[1], [2.5], [3.1], [4.6], [5.2], [6.7], [7.3]]
    labels = [3000, 4500, 5100, 6600, 7200, 8700, 9300]
    return input_vecs, labels

def train_linear_unit():
    '''
    Trainning linear unit
    '''
    # Build linear unit with one param which is work experience and active function
    linearUnit = LinearUnit(1, active_func)
    # Trainning with 100 iteration and learn rate
    input_vecs, labels = get_training_dataset()
    linearUnit.train(input_vecs, labels, 100, 0.01)
    # Return linear unit trainned
    return linearUnit

def draw_figure(result):
    '''
    Draw figure
    '''
    plt.figure()

    X = list(result.keys())
    Y = list(result.values())
    _Y = [1000*x+2000 for x in X]
    plt.plot(X, _Y, color="black", linewidth = 2, label="y=1000x+2000")
    T = np.arctan2(X, Y)
    S = 20
    plt.scatter(X, Y, c=T, s=S, marker='o')
    plt.title("WORK EXPERIENCE & SALATY")
    plt.xlabel("Work years(YEAR)")
    plt.ylabel("Salary(YUAN)")
    plt.xticks(range(16))
    # plt.grid(True, linestyle="-", color="gray")
    plt.legend()
    plt.autoscale()
    plt.show()


if __name__ == '__main__': 
    '''
    Trainning linear unit
    '''
    linear_unit = train_linear_unit()
    # Print weight
    print (linear_unit)
    # Check
    items = [[1], [2.5], [3.1], [4.6], [5.2], [6.7], [7.3], [8.8], [9.4], [10], [11], [12], [13], [14], [15]]
    result = { k[0] : linear_unit.predict(k) for k in items}

    for (k, v) in result.items():
        print('Work %d years, monthly salary = %.2f' % (k, v))

    # Draw figure
    draw_figure(result)