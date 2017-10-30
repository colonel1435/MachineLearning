#!usr/bin/env python3.6
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Zero
# #  Description : 
# #  Time        : 2017/10/30 0030

import numpy as np

class ReluActivitor():
    def forward(self, input_weight):
        return max(0, input_weight)

    def backward(self, output):
        return 1 if (output > 0) else 0

class ConvKernel():
    '''
    '''
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-1, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

class ConvLayer():
    '''
    '''
    def __init__(self, input_width, input_height,channel_number,
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
        self.activitor = activator
        self.learning_rate = learning_rate
        self.output_width = ConvLayer.calculate_ouput_size(input_width, kernel_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_ouput_size(input_height, kernel_height, zero_padding, stride)
        self.output_array = np.zeros((self.kernel_number, self.output_height, self.output_width))
        self.kernels = []
        for i in range(kernel_number):
            self.kernels.append(ConvKernel(kernel_width, kernel_height, self.channel_number))

    @staticmethod
    def calculate_ouput_size(input_size, kernel_size, zero_padding, stride):
        return (input_size - kernel_size + 2 * zero_padding) / stride + 1

    def element_wise_on(self, array, op):
        for i in np.nditer(array, op_flags=['readwrite']):
            i[...] = op(i)

    def conv(self, input_array, kernel_array, output_array, stride, bias):
        channer_number = input_array.ndim
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

    def padding(self, input_array, zp):
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