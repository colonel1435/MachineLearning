#!usr/bin/env python3.6
# -*- coding: utf-8 -*-
# #  FileName    : 
# #  Author      : Zero
# #  Description :  HMM algorithm, Refer to http://www.52nlp.cn/category/hidden-markov-model
# #  Time        : 2017/10/25 0025

import numpy as np


class HMM:
    '''
    A:  状态转移矩阵A[1..N][1..N]. a[i][j] 是从t时刻状态i到t+1时刻状态j的转移概率 
    B:  混淆矩阵B[1..N][1..M]. b[j][k]在状态j时观察到符合k的概率
    Pi: 初始向量pi[1..N]，pi[i] 是初始状态概率分布
    O:  观察序列
    M:  观察符号数目; V={1,2,...,M}
    N:  隐藏状态数目; Q={1,2,...,N}
    '''
    def __init__(self, A, B, Pi, O):
        self.A = np.array(A, np.float)
        self.B = np.array(B, np.float)
        self.Pi = np.array(Pi, np.float)
        self.O = np.array(O, np.float).astype(int)
        self.M = self.B.shape[1]
        self.N = self.A.shape[0]

        print(self.A)
        print(self.B)
        print(self.Pi)
        print(self.O)
        print(self.M)
        print(self.N)
    def forward(self):
        '''
        Finding the probability of an observed sequence
        '''

        # Length of observed sequence
        T = len(self.O)
        # Partial probability
        alpha = np.zeros((T, self.N), np.float)
        # Init probability when T = 0, P = pi[i] * B[i][0]
        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]

        # Compute probability when T > 0 with recursion; alpha(t+1)[i] = sum(alpha[t][j]*A[j][i]) * B[i][O(t+1)]
        for t in range(T - 1):
            for i in range(self.N):
                tmp_sum = 0
                for j in range(self.N):
                    tmp_sum += alpha[t, j] * self.A[j, i]
                alpha[t+1, i] = tmp_sum * self.B[i, self.O[t + 1]]
        # Sum probability when t = T
        prob = 0
        for i in range(self.N):
            prob += alpha[T - 1, i]

        return prob, alpha

    def viterbi(self):
        '''
        Finding most probable sequence of hidden state
        '''
        # Length of observed sequence
        T = len(self.O)
        # Status sequence
        I = np.array(T, np.float)
        delta = np.zeros((T, self.N), np.float)
        psi = np.zeros((T, self.N), np.float)
        # Init probability and path pointer when T = 0, P = pi[i] * B[i][0]
        for i in range(self.N):
            delta[0][i] = self.Pi[i] * self.B[i][self.O[0]]
            psi[0, i] = 0
        # Compute max probability when T > 0
        for t in range(1, T):
            for i in range(self.N):
                delta[t][i] = self.B[i][self.O[t]] * np.array([delta[t - 1, j] * self.A[j, i]
                                                               for j in range(self.N)]).max()
                psi[t, i] = np.array( [delta[t - 1, j] * self.A[j, i]
                                       for j in range(self.N)]).argmax()

        I[T - 1] = delta[T - 1, :].argmax()
        # Compute max probability when T > 0
        for t in range(T - 2, -1, -1):
            I[t] = psi[t + 1, I[t+1]]

        return I

def test_hmm_forward():
    A = np.array([
        [0.33, 0.33, 0.33],
        [0.33, 0.33, 0.33],
        [0.33, 0.33, 0.33]
    ], dtype=np.float)
    B = np.array([
        [0.5, 0.5],
        [0.75, 0.25],
        [0.25, 0.75]
    ], dtype=np.float)
    Pi = np.array([0.33, 0.33, 0.33], np.float)
    O = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], np.int)
    hmm = HMM(A, B, Pi, O)
    ret = hmm.forward()
    print("Probability -> {0}\nAlpha ->\n {1}".format(ret[0], ret[1]))

def test_hmm_viterbi():
    A = np.array([
        [0.33, 0.33, 0.33],
        [0.33, 0.33, 0.33],
        [0.33, 0.33, 0.33]
    ], dtype=np.float)
    B = np.array([
        [0.5, 0.5],
        [0.75, 0.25],
        [0.25, 0.75]
    ], dtype=np.float)
    Pi = np.array([0.33, 0.33, 0.33], np.float)
    O = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    hmm = HMM(A, B, Pi, O)
    print("Hiden state -> {0}".format(hmm.viterbi()))
if __name__ == '__main__':
    # test_hmm_forward()
    test_hmm_viterbi()