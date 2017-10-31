#!usr/bin/env python3.6
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Zero
# #  Description : 
# #  Time        : 2017/10/30 0030

import numpy as np
from Activator import IdentityActivator, ReluActivator

def get_patch(input_array, i, j, filter_width,
              filter_height, stride):
    '''
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]

def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth, input_height + 2 * zp, input_width + 2 * zp
            ))
            padded_array[:,
                zp:zp+input_height, zp : zp + input_width] = input_array

            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp, input_width + 2 * zp
            ))
            padded_array[zp:zp + input_height, zp: zp + input_width] = input_array

            return padded_array

def conv(input_array, kernel_array, output_array, stride, bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width, kernel_height, stride)
                * kernel_array
            ).sum() + bias

class ConvKernel():
    '''
    '''
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-1, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

class ConvLayer():
    '''
    '''
    def __init__(self, input_width, input_height, channel_number,
                 kernel_width, kernel_height, kernel_number,
                 zero_padding, stride,
                 activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_number = kernel_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.activator = activator
        self.learning_rate = learning_rate
        self.output_width = ConvLayer.calculate_ouput_size(input_width, kernel_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_ouput_size(input_height, kernel_height, zero_padding, stride)
        self.output_array = np.zeros((self.kernel_number, self.output_height, self.output_width))
        self.kernels = []
        for i in range(kernel_number):
            self.kernels.append(ConvKernel(kernel_width, kernel_height, self.channel_number))

    @staticmethod
    def calculate_ouput_size(input_size, kernel_size, zero_padding, stride):
        return int((input_size - kernel_size + 2 * zero_padding) / stride + 1)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        expand_array = self.expand_sensitivity_map(sensitivity_array)
        expand_width = expand_array.shape[2]
        zp =  int((self.input_width + self.kernel_width - 1 - expand_width) / 2)
        padded_array = padding(expand_array, zp)
        self.delta_array = self.create_delta_array()
        for f in range(self.kernel_number):
            kernel = self.kernels[f]
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                kernel.get_weights()
            ))

            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        expand_width = (self.input_width - self.kernel_width + 2 * self.zero_padding + 1)
        expand_height = (self.input_height - self.kernel_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expand_height, expand_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    def bp_gradient(self, sensitivity_array):
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.kernel_number):
            filter = self.kernels[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
                filter.bias_grad = expanded_array[f].sum()

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.kernel_number):
            kernel = self.kernels[f]
            conv(self.padded_input_array,
                 kernel.get_weights(), self.output_array[f],
                 self.stride, kernel.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        for kernel in self.kernels:
            kernel.update(self.learning_rate)


def init_test():
    a = np.array([
        [[0,1,1,0,2],
         [2,2,2,2,1],
         [1,0,0,0,0],
         [0,1,1,0,0],
         [1,2,0,0,2]],
        [[1,0,2,2,0],
         [0,0,0,2,0],
         [1,2,1,2,1],
         [1,0,0,0,0],
         [1,2,1,1,1]],
        [[2,1,2,0,0],
         [1,0,0,1,0],
         [0,2,1,0,1],
         [0,1,2,2,2],
         [2,1,0,0,1]]
    ])

    b = np.array([
        [[0, 1, 1],
         [2, 2, 2],
         [1, 0, 0]],
        [[1, 0, 2],
         [0, 0, 0],
         [1, 2, 1]]
    ])

    c1 = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    c1.kernels[0].weights = np.array(
        [[
            [-1, 1, 0],
            [0, 1, 0],
            [0, 1, 1]
        ],
         [
             [-1, -1, 0],
             [0, 0, 0],
             [0, -1, 0]
         ],
         [
             [0, 0, -1],
             [0, 1, 0],
             [1, -1, -1]
         ]
        ], dtype=np.float64)
    c1.kernels[0].bias = 1
    c1.kernels[1].weights = np.array(
        [[
            [1, 1, -1],
            [-1, -1, 1],
            [0, 1, 0]
        ],
            [
                [0, 1, 0],
                [-1, 0, -1],
                [-1, 1, 0]
            ],
            [
                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 0, 0]
            ]
        ], dtype=np.float64)

    return a, b, c1


def gradient_check():
    error_func = lambda o: o.sum()

    a, b, c1 = init_test()
    c1.forward(a)
    sensitivity_array = np.zeros(c1.output_array.shape,
                                 dtype=np.float64)
    c1.backward(a, sensitivity_array, IdentityActivator())
    epsilon = 10e-4
    for d in range(c1.kernels[0].weights_grad.shape[0]):
        for i in range(c1.kernels[0].weights_grad.shape[1]):
            for j in range(c1.kernels[0].weights_grad.shape[2]):
                c1.kernels[0].weights[d, i, j] += epsilon
                c1.forwar(a)
                err1 = error_func(c1.output_array)
                c1.kernels[0].weights[d, i, j] -= 2*epsilon
                c1.forward()
                err2 = error_func(c1.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                c1.kernels[0].weights[d, i, j] += epsilon
                print('weights({%d},{%d},{%d}); expected({%f}); actural({%f})'.format(
                    d, i, j, expect_grad, c1.kernels[0].weights_grad[d, i, j]
                ))


if __name__ == '__main__':
    gradient_check()

