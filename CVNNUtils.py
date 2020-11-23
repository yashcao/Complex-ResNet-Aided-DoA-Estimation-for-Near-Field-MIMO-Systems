# import numpy as np
import tensorflow as tf
# from tensor2tensor.layers import common_layers
from numpy import pi
import CVNNLayer
# from sklearn.utils import shuffle


def Block_1D(inpt, features, filters):
    conv1 = CVNNLayer.conv1d(
        inpt,
        weightShape=(3, features, filters),
        biasDim=filters,
        convStride=1,
        convPadding='SAME')
    conv1 = CVNNLayer.phase_amplitude(conv1)
    # conv1 = CVNNLayer.cTanh(conv1)

    conv2 = CVNNLayer.conv1d(
        conv1,
        weightShape=(3, filters, filters),
        biasDim=filters,
        convStride=1,
        convPadding='SAME')
    conv2 = CVNNLayer.phase_amplitude(conv2)
    # conv2 = CVNNLayer.cTanh(conv2)

    if int(inpt.r.shape[-1]) != filters:
        inpt = CVNNLayer.conv1d(
            inpt,
            weightShape=(1, features, filters),
            biasDim=filters,
            convStride=1,
            convPadding='SAME')

    res = CVNNLayer.add_Complex(conv2, inpt)
    return res


def TASKs(x, out_nums, keep_prob=None, mu=0, sigma=0.1, networkType='Complex'):

    if networkType is not 'Real':
        x = CVNNLayer.typeClone(x, networkType)

    # Input = 32x1. Output = 8x8.
    net = Block_1D(x, 1, 16)
    net = CVNNLayer.maxpool(net, size=[2], strides=[2], padding='SAME')
    net = Block_1D(net, 16, 8)
    net = CVNNLayer.maxpool(net, size=[2], strides=[2], padding='SAME')
    # net = CVNNLayer.normalize(net)

    # Output = 2x4.
    net = Block_1D(net, 8, 4)
    net = CVNNLayer.maxpool(net, size=[2], strides=[2], padding='SAME')
    net = Block_1D(net, 4, 4)
    net = CVNNLayer.maxpool(net, size=[2], strides=[2], padding='SAME')
    # net = CVNNLayer.normalize(net)

    # net_drop = tf.nn.dropout(net, 0.5)  # 模型小的话不采用
    # net = CVNNLayer.global_avg_pool(net)  # 大数据类别分类采用

    return net


def task_1d(net, out_nums=1, keep_prob=None):
    Input_num = 8
    fc0 = CVNNLayer.reshape_cp(net, [-1, Input_num])

    # Fully Connected. Input = 50x128. Output = 50x32. fc1.r.shape[-1]=32
    fc1 = CVNNLayer.affine(
        fc0, weightShape=(Input_num, 10), biasDim=10, kp=keep_prob)
    fc1 = CVNNLayer.cTanh(fc1)

    # Input = 50x32. Output = 50x10.
    fc2 = CVNNLayer.affine(
        fc1, weightShape=(10, out_nums), biasDim=out_nums, kp=keep_prob)

    fc2 = CVNNLayer.Phase(fc2)  # 归一化幅度，虚部即正弦值
    logits = CVNNLayer.reshape_cp(fc2, (-1, out_nums))

    return logits
