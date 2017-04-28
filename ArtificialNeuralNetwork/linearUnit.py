#!usr/bin/env python3.6.1
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Mr.wumin
# #  Description : Refer to https://www.zybuluo.com/hanbingtao/note/448086
# #  Time        : 2017/4/21

from MachineLearning.ArtificialNeuralNetwork.perceptron import Perceptron

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

if __name__ == '__main__': 
    '''
    Trainning linear unit
    '''
    linear_unit = train_linear_unit()
    # Print weight
    print (linear_unit)
    # Check
    print('Work 1 years, monthly salary = %.2f' % linear_unit.predict([1]))
    print('Work 2.5 years, monthly salary = %.2f' % linear_unit.predict([2.5]))
    print('Work 3.1 years, monthly salary = %.2f' % linear_unit.predict([3.1]))
    print ('Work 4.6 years, monthly salary = %.2f' % linear_unit.predict([4.6]))
    print ('Work 5.2 years, monthly salary = %.2f' % linear_unit.predict([5.2]))
    print ('Work 6.7 years, monthly salary = %.2f' % linear_unit.predict([6.7]))
    print ('Work 7.3 years, monthly salary = %.2f' % linear_unit.predict([7.3]))