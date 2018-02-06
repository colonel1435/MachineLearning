#!usr/bin/env python3.6.1
# -*- coding: utf-8 -*-
# #  FileName    :
# #  Author      : Mr.wumin
# #  Description : Refer to https://www.zybuluo.com/hanbingtao/note/433855
# #  Time        : 2017/4/21
from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        Init active function, weight, bias
        '''
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        '''
        Print weight and bias
        '''
        return 'weights\t:%s\tbias\t:%f\n' % (self.weights, self.bias)


    def predict(self, input_vec):
        '''
        Return the compute result of the perceptron according to input vector
        '''

        # First, package input_vec[x1,x2,x3,...] and weights[w1,w2,w3,...] to list like [(x1,w1),(x2,w2),(x3,w3),...]
        # Then, compute [x1*w1, x2*w2, x3*w3, ...] with mao()
        # Finally, sum with reduce()
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda x_w: x_w[0] * x_w[1],
                       zip(input_vec, self.weights))
                , 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        Start trainning with input vectors, labels, number of iteration and learn rate
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        Iter input_vecs to train the whole data
        '''
        # Package input_vec[x1,x2,x3,...] and label[y1, y2, y3,...] to list like ((x1,y1),(x2,y2),(x3,y3),...)
        samples = zip(input_vecs, labels)
        # Compute the next output then update weight
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        Update weight with new out and rate
        '''
        # Package input_vec[x1,x2,x3,...] and weights[w1,w2,w3,...] to list like [(x1,w1),(x2,w2),(x3,w3),...]
        # Then update weight and bias
        delta = label - output
        self.weights = list(map(
            lambda x_w: x_w[1] + rate * delta * x_w[0],
            zip(input_vec, self.weights)))
        self.bias += rate * delta

def active_func(x):
    '''
    Define active function
    '''
    return 1 if x > 0 else 0

def get_training_dataset():
    '''
    Build trainning dataset based on true table
    '''
    # input data
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # ture table of and function
    labels = [1, 0, 0, 0]
    # ture table of or function
    # labels = [1, 0, 1, 1]
    return input_vecs, labels

def train_perceptron():
    '''
    Trainning perceptron
    '''
    # Build perceptron with 2 input params and active function
    p = Perceptron(2, active_func)
    # Trainning with iteration, learn rate params
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # Return the perceptron trainned
    return p


if __name__ == '__main__':
    # Trainning and perceptron
    perception = train_perceptron()
    # Print params of perceptron trainned
    print (perception)
    # Check
    print ('1 and 1 = %d' % perception.predict([1, 1]))
    print ('0 and 0 = %d' % perception.predict([0, 0]))
    print ('1 and 0 = %d' % perception.predict([1, 0]))
    print ('0 and 1 = %d' % perception.predict([0, 1]))
		