#!usr/bin/env python3.6
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Zero
# #  Description : 
# #  Time        : 2017/10/31 0031
import numpy as np

class SigmoidActivator():
    ''''''
    def forward(self, input):
        return 1.0 / (1.0 + np.exp(- input))

    def backward(self, output):
        return output * (1 - output)


class ReluActivator():
    def forward(self, input_weight):
        return max(0, input_weight)

    def backward(self, output):
        return 1 if (output > 0) else 0


class IdentityActivator():
    def forward(self, input_weight):
        return input_weight

    def backward(self, output):
        return 1