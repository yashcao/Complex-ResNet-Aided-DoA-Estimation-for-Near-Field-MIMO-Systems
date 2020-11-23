# Front matter: load libraries needed for training
import numpy as np
from numpy import random
from numpy import linalg as la
import tensorflow as tf
from scipy.linalg import toeplitz
# import matplotlib.pyplot as plt
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
# plt.switch_backend('agg')
from scipy import linalg

N = 65  # 2p+1; p=32
L = 1  # 2+1
snr = 10  # signal-noise-ratio
# print(snr)
K = 100  # snap shot

dd = 0.5  # dd = d/lambda; d=lamda/2
lamda = 0.0107  # 5mm in（0mm，10mm） 0.1
d = dd * lamda

rad = np.pi / 180
drad = 180 / np.pi
R = [[1000 * lamda]]  # 5 m

Theta = np.array(np.linspace(-40, 60, 101)).reshape(101, 1)
Theta = Theta * rad

n_d = np.arange(1, N + 1, 1)
n_d = n_d - [33]  # N rows

sim_num = len(Theta)
mts = 10
# mmse = np.zeros([mts])
# pre_X = np.zeros([sim_num, 65, 1], dtype=complex)

data_length = 33

pre_X = np.zeros([sim_num, data_length, 1], dtype=complex)
pre_y = np.zeros([sim_num, 1])

R_new = np.zeros([N, N], dtype=complex)


def net(pre_x):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Predicting...")

        signature_key = 'test_signature'
        input_key = 'input_x'
        output_key = 'output'

        meta_graph_def = tf.saved_model.loader.load(
            sess, ['test_saved_model'], "./saveModel_32/")
        # meta_graph_def = tf.saved_model.loader.load(
        #     sess, ['test_saved_model'], "E:/Documents/Python/cys_8/plot-1d-new/saveModel/")

        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        # _x 实际输入待inference的data
        y_p = sess.run(y, feed_dict={x: pre_x})
        pre_y = y_p * drad
        return pre_y


def GenArrayVector(source_theta, R_d):
    '''
    source_theta is the selected row of
    Theta, consists of L angles.
    '''
    d0 = R_d

    A = np.zeros((N, L), dtype=complex)

    for i in range(0, L):
        # print(source_theta[i])
        # print(d0[i])
        d_r = d0[i] * np.sqrt(1 + (n_d * d / d0[i])**2 -
                              (2 * n_d * d * np.sin(source_theta[i]) / d0[i]))
        a_r = np.exp(-2j * np.pi * (d_r - d0[i]) / lamda) * d0[i] / d_r  # Nx1
        # print(a_r)
        A[:, i] = a_r

    return A


def GenArray(source_theta, d0):
    # source_theta = np.array(source_theta)

    d_r = d0 * np.sqrt(1 + (n_d * d / d0)**2 -
                       (2 * n_d * d * np.sin(source_theta) / d0))
    # print(d_r)
    a_r = np.exp(-2j * np.pi * (d_r - d0) / lamda) * d0 / d_r  # Nx1
    # a_r = np.array(a_r).reshape(N, 1)
    # print(a_r.shape)

    return a_r


# 发送BPSK信号/训练序列 S
# f0 = 60e3  # 60kHz

# print(S)
# print(np.dot(S, S.T))
var = 1  # var 表示方差


def GenSignal(A, S):
    # 近似协方差
    R_y = np.zeros((N, N))

    noise = np.sqrt(var / 2) * (random.randn(N, K) + 1j * random.randn(N, K))

    y_temp = np.dot(A, S)
    # y_0[:, sim] = y_temp.reshape(65, )

    # 接收信号 y
    y_s = y_temp + noise
    y_s_t = y_s.T.conj()
    R_y = np.matmul(y_s, y_s_t) / K

    return R_y


def ReduceData(R_xx, row):
    '''
    取上三角元素
    '''
    vector = []
    for i in range(0, row):
        a = R_xx[i, i + 1:]
        vector = np.append(vector, a)

    # b = R_xx.diagonal()
    # b = b[1:]

    # vector = np.append(a, b)

    return vector


def DiagData(R_xx, row):
    '''
    取反对角线元素
    '''
    # vector = []
    vector = np.rot90(R_xx)
    vector = np.diag(vector)  # (65,)

    return vector


def gen_toeplitz(m, array):
    '''
    m:     中心项索引
    array: 数列
    '''
    # N = 2 * m + 1       # 总共的项数

    list2 = array[::-1]  # 翻转

    t_matrix = toeplitz(array[m:], list2[m:])
    return t_matrix


def diag_opreation(k, i, Num_element, R2, A1):
    c = int((Num_element-1)/2)
    while np.max([k, i])<Num_element:
        # print(k, i)
        if np.abs(k-i)%2==0:
            R2[k, i] = A1[c-int((i-k)/2), c+int((i-k)/2)]
        else:
            R2[k, i] = (A1[c-int((i-k-1)/2), c+int((i-k+1)/2)]+A1[c-int((i-k+1)/2), c+int((i-k-1)/2)])/2
        k=k+1
        i=i+1
    return R2



for mt in range(0, mts):
    print("predicting: " + str(mt))
    S = np.sqrt(0.5) * (random.randn(L, K) + 1j * random.randn(L, K))
    S = np.sqrt(10**(snr / 10)) * S
    for ki in range(sim_num):
        A = GenArrayVector(Theta[ki], R)
        R_y = GenSignal(A, S)

        for j in range(N):
            # print()
            i = 0
            # print(i, j)
            R_new = diag_opreation(i, j, N, R_new, R_y)

        for k in range(1, N):
            # print()
            i = 0
            # print(i, j)
            R_new = diag_opreation(k, i, N, R_new, R_y)

        R_new_re = R_new[(N-33)//2:N-((N-33)//2), (N-33)//2:N-((N-33)//2)]
        U, sigma, VT = la.svd(R_new_re)
        Usig = U[:, 0:L]
        # Usig = np.reshape(R_new[-1], (N * L))

        pre_X[ki, :] = Usig

    pre_X_re = np.reshape(pre_X, (sim_num, data_length*L))
    norm_tr = np.reshape(pre_X_re[:,0], (sim_num,1))*np.ones((1,data_length))
    pre_X_re = pre_X_re/norm_tr
    #pre_X = np.array(linalg.orth(pre_X))
    pre_X_re = np.delete(pre_X_re, 0, axis=1)
    pre_X_re = np.reshape(pre_X_re, (sim_num, data_length*L-1, 1))

    # pre_X = pre_X/np.abs(pre_X)
    pre_y = net(pre_X_re) + pre_y
    # mmse[mt] = np.sqrt(np.square(pre_y - Theta * drad)) + mmse[mt]


# mmse = mmse / mts
# Theta = Theta * drad
pre_y = pre_y/mts

# np.save("net-snr-1.npy", snr)
np.save("cvnn_doa.npy", pre_y)


'''
plt.figure()

plt.plot(snr, mmse / mts)

# my_x_ticks = np.arange(-90, 90, 15)
# my_y_ticks = np.arange(10*lamda, 90 * lamda, 10)

# plt.xticks(my_x_ticks)
# plt.yticks(y_ticks, (y_ticks / lamda).astype(np.int))
plt.xlabel('snapshots')
plt.ylabel('MMSE')
# plt.title('MUSIC')

plt.show()
'''
