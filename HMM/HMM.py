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

        print("A ->\n{0}".format(self.A))
        print("B ->\n{0}".format(self.B))
        print("Pi ->\n{0}".format(self.Pi))
        print("O ->\n{0}".format(self.O))
        print("M ->\n{0}".format(self.M))
        print("N ->\n{0}".format(self.N))

    def forward(self):
        '''
        Finding the probability of an observed sequence
        '''

        # Length of observed sequence
        T = len(self.O)
        # Partial probability
        alpha = np.zeros((T, self.N), np.float)
        # Init probability when t = 0, P = pi[i] * B[i][0]
        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]

        # Compute probability when t > 0 with recursion; alpha(t+1)[i] = sum(alpha[t][j]*A[j][i]) * B[i][O(t+1)]
        for t in range(T - 1):
            for i in range(self.N):
                tmp_sum = 0
                for j in range(self.N):
                    tmp_sum += alpha[t, j] * self.A[j, i]
                alpha[t+1, i] = tmp_sum * self.B[i, self.O[t + 1]]
        # Sum probability when t = T
        alpha_prob = 0.0
        for i in range(self.N):
            alpha_prob += alpha[T - 1, i]

        return alpha_prob, alpha

    def viterbi(self):
        '''
        Finding most probable sequence of hidden state
        '''
        # Length of observed sequence
        T = len(self.O)
        # Status sequence
        I = np.zeros(T, np.int)
        # Partial probability
        delta = np.zeros((T, self.N), np.float)
        # Reverse pointer
        psi = np.zeros((T, self.N), np.float)
        # Init probability and path pointer when t = 0, P = pi[i] * B[i][0]
        for i in range(self.N):
            delta[0][i] = self.Pi[i] * self.B[i][self.O[0]]
            psi[0, i] = 0
        # Compute max probability when t > 0
        for t in range(1, T):
            for i in range(self.N):
                delta[t][i] = self.B[i][self.O[t]] * np.array([delta[t - 1, j] * self.A[j, i]
                                                               for j in range(self.N)]).max()
                psi[t, i] = np.array( [delta[t - 1, j] * self.A[j, i]
                                       for j in range(self.N)]).argmax()

        I[T - 1] = delta[T - 1, :].argmax()
        # Compute max probability when t > 0
        for t in range(T - 2, -1, -1):
            I[t] = psi[t + 1, I[t + 1]]

        return I, delta

    def backward(self):
        '''
        Generating a HMM from a sequence of obersvations
        '''
        # Length of observed sequence
        T = len(self.O)
        # Partial probability
        beta = np.zeros((T, self.N), np.float)
        # Init probability when t = T, P = pi[i] * B[i][0]
        for i in range(self.N):
            beta[T - 1, i] = 1.0

        # Compute probability when t < T with recursion; bata(t)[i] = sum(A[i][j]) * B[i][O(t-1)]*bata[t+1][j])
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                tmp_sum = 0.0
                for j in range(self.N):
                    tmp_sum += self.B[i, self.O[t - 1]] * self.A[i, j] * beta[t + 1, j]
                beta[t, i] = tmp_sum

        # Sum probability when t = 0
        beta_prob = 0.0
        for i in range(self.N):
            beta_prob += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]

        return beta_prob, beta

    def gamma(self, alpha, beta):
        '''
         Expect number of occurrence q(i)
        '''
        T = len(self.O)
        gamma = np.zeros((T, self.N), np.float)
        for t in range(T):
            for i in range(self.N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / \
                              sum(alpha[t, j] * beta[t, j] for j in range(self.N))

        return gamma

    def xi(self, alpha, beta):
        '''
        Expect number of translation fron q(i) to q(j)
        '''
        T = len(self.O)
        xi = np.zeros((T - 1, self.N, self.N), np.float)
        for t in range(T - 1):
            for i in range(self.N):
                for j in range(self.N):
                    numerator = alpha[t, i] * self.A[i, j] * self.B[j, self.O[t + 1]] * beta[t + 1, j]
                    denominator = sum(sum(
                        alpha[t, ii] * self.A[ii, jj] * self.B[jj, self.O[t + 1]] * beta[t + 1, jj]
                            for ii in range(self.N))
                                      for jj in range(self.N))
                    xi[t, i, j] = numerator / denominator

        return xi

    def baum_welch(self):
        '''
        Baum Welch algorithm
        '''
        T = len(self.O)
        V = [k for k in range(self.M)]

        # Init lambda. such as A, B, Pi
        self.A = np.array(([0, 1, 0], [0.4, 0, 0.6], [0, 0.5, 0.5]), np.float)
        self.B = np.array(([0.5, 0.5], [0.3, 0.7], [0.6, 0.4]), np.float)
        # self.A = np.array([[1.0 / self.N] * self.N] * self.N) # must array back, then can use[i,j]
        # self.B = np.array([[1.0 / self.M] * self.M] * self.N)
        self.Pi = np.array(([1.0 / self.N] * self.N), np.float)

        x = 1
        delta_lambda = x + 1
        times = 0
        # iteration - lambda
        while delta_lambda > x:
            _, alpha = self.forward()
            _, beta = self.backward()
            gamma = self.gamma(alpha, beta)
            xi = self.xi(alpha, beta)
            lambda_n = [self.A, self.B, self.Pi]

            '''
            A[i, j] = num(A(t, i, j)) / num(A(t, i))
            '''
            for i in range(self.N):
                for j in range(self.N):
                    numerator = sum(xi[t, i, j] for t in range(T - 1))
                    denominator = sum(gamma[t, i] for t in range(T - 1))
                    self.A[i, j] = numerator / denominator
            '''
            B[i, j] = num(O(t, k)) / num(A(t, j))
            '''
            for j in range(self.N):
                for k in range(self.M):
                    numerator = sum(gamma[t, j] for t in range(T) if self.O[t] == V[k])
                    denominator = sum(gamma[t, j] for t in range(T))
                    self.B[j, k] = numerator / denominator
            '''
            Pi[i] = gamma[i]
            '''
            for i in range(self.N):
                self.Pi[i] = gamma[0, i]

            delta_A = map(abs, lambda_n[0] - self.A)
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi)])
            times += 1
            print("Times -> {0} delta_lamba -> {1}".format(times, delta_lambda))

        return self.A, self.B, self.Pi

def test_hmm_forward():
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ], dtype=np.float)
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ], dtype=np.float)
    Pi = np.array([0.2, 0.4, 0.4], np.float)
    O = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], np.int)
    hmm = HMM(A, B, Pi, O)
    ret = hmm.forward()
    print("Probability -> {0}\nAlpha ->\n {1}".format(ret[0], ret[1]))

def test_hmm_viterbi():
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ], dtype=np.float)
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ], dtype=np.float)
    Pi = np.array([0.2, 0.4, 0.4], np.float)
    O = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    hmm = HMM(A, B, Pi, O)
    ret = hmm.viterbi()
    print("Hiden state -> {0}\nAlpha -> \n{1}".format(ret[0], ret[1]))

def test_hmm_backward():
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ], dtype=np.float)
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ], dtype=np.float)
    Pi = np.array([0.2, 0.4, 0.4], np.float)
    O = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    hmm = HMM(A, B, Pi, O)
    ret = hmm.backward()
    print("Probability -> {0}\nBeta ->\n {1}".format(ret[0], ret[1]))

def test_baum_welch():
    O = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    hmm = HMM(np.zeros((3, 3)), np.zeros((3, 2)), np.array(3), O)
    lambda_em = hmm.baum_welch()
    print("A ->\n{0}\nB ->\n{1}\nPi -> \n{2}".format(lambda_em[0], lambda_em[1], lambda_em[2]))


if __name__ == '__main__':
    # test_hmm_forward()
    # test_hmm_viterbi()
    # test_hmm_backward()
    test_baum_welch()
