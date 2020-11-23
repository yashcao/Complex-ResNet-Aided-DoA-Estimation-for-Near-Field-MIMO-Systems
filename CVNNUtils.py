# import numpy as np
import tensorflow as tf
# from tensor2tensor.layers import common_layers
from numpy import pi
import CVNNLayer
# from sklearn.utils import shuffle
# from LSTMLayer import LSTM_RNN


'''
def evaluateAcc(X_data, y_data):

    num_examples = len(X_data)
    sess = tf.get_default_session()
    total_accuracy =
    return total_accuracy / num_examples
'''


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



def Block_2D(inpt, features, filters):
    conv1 = CVNNLayer.conv2d(
        inpt,
        weightShape=([3, 3, features, filters]),  # w,h,in_c,out_c
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv1 = CVNNLayer.zRelu(conv1)

    conv2 = CVNNLayer.conv2d(
        conv1,
        weightShape=([3, 3, filters, filters]),
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv2 = CVNNLayer.zRelu(conv2)

    conv3 = CVNNLayer.conv2d(
        conv2,
        weightShape=([3, 3, filters, filters]),
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv3 = CVNNLayer.zRelu(conv3)

    if int(inpt.r.shape[-1]) != filters:
        inpt = CVNNLayer.conv2d(
            inpt,
            weightShape=([1, 1, features, filters]),
            biasDim=filters,
            convStride=[1, 1, 1, 1],
            convPadding='SAME')

    res = CVNNLayer.add_Complex(conv3, inpt)
    return res


def Block_1D_flow(inpt, features, filters):
    # batch = inpt.r.get_shape().as_list()[0]
    conv11 = CVNNLayer.conv1d(
        inpt,
        weightShape=(3, features, filters),
        biasDim=filters,
        convStride=1,
        convPadding='SAME')
    conv11 = CVNNLayer.cTanh(conv11)
    # conv11 = CVNNLayer.phase_amplitude(conv11)

    conv12 = CVNNLayer.conv1d(
        inpt,
        weightShape=(3, features, filters),
        biasDim=filters,
        convStride=1,
        convPadding='SAME')
    conv12 = CVNNLayer.cTanh(conv12)

    conv1 = CVNNLayer.add_Complex(conv11, conv12)

    conv21 = CVNNLayer.conv1d(
        conv1,
        weightShape=(3, filters, filters),
        biasDim=filters,
        convStride=1,
        convPadding='SAME')
    conv21 = CVNNLayer.cTanh(conv21)

    conv22 = CVNNLayer.conv1d(
        conv1,
        weightShape=(3, filters, filters),
        biasDim=filters,
        convStride=1,
        convPadding='SAME')
    conv22 = CVNNLayer.cTanh(conv22)

    conv2 = CVNNLayer.add_Complex(conv21, conv22)

    if int(inpt.r.shape[-1]) != filters:
        inpt = CVNNLayer.conv1d(
            inpt,
            weightShape=(1, features, filters),
            biasDim=filters,
            convStride=1,
            convPadding='SAME')

    res = CVNNLayer.add_Complex(conv2, inpt)
    return res


def Block_2D_flow(inpt, features, filters):
    conv11 = CVNNLayer.conv2d(
        inpt,
        weightShape=([3, 3, features, filters]),  # w,h,in_c,out_c
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv11 = CVNNLayer.zRelu(conv11)

    conv12 = CVNNLayer.conv2d(
        inpt,
        weightShape=([3, 3, features, filters]),  # w,h,in_c,out_c
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv12 = CVNNLayer.zRelu(conv12)

    conv1 = CVNNLayer.add_Complex(conv11, conv12)

    conv21 = CVNNLayer.conv2d(
        conv1,
        weightShape=([3, 3, filters, filters]),
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv21 = CVNNLayer.zRelu(conv21)

    conv22 = CVNNLayer.conv2d(
        conv1,
        weightShape=([3, 3, filters, filters]),
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv22 = CVNNLayer.zRelu(conv22)

    conv2 = CVNNLayer.add_Complex(conv21, conv22)

    conv31 = CVNNLayer.conv2d(
        conv2,
        weightShape=([3, 3, filters, filters]),
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv31 = CVNNLayer.zRelu(conv31)

    conv32 = CVNNLayer.conv2d(
        conv2,
        weightShape=([3, 3, filters, filters]),
        biasDim=filters,
        convStride=[1, 1, 1, 1],
        convPadding='SAME')
    conv32 = CVNNLayer.zRelu(conv32)

    conv3 = CVNNLayer.add_Complex(conv31, conv32)

    if int(inpt.r.shape[-1]) != filters:
        inpt = CVNNLayer.conv2d(
            inpt,
            weightShape=([1, 1, features, filters]),
            biasDim=filters,
            convStride=[1, 1, 1, 1],
            convPadding='SAME')

    res = CVNNLayer.add_Complex(conv3, inpt)
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


def DNN_Net(x, out_nums, keep_prob, mu=0, sigma=0.1, networkType='Complex'):

    # if networkType is not 'Real':
    # x = layersHDA.typeClone(x, networkType)
    # x = layersHDA.normalize(x)

    net = CVNNLayer.affine(x, weightShape=(130, 30), biasDim=30)
    net = CVNNLayer.cTanh(net)

    net = CVNNLayer.affine(net, weightShape=(30, 10), biasDim=10)
    net = CVNNLayer.cTanh(net)

    # net = layersHDA.affine(net, weightShape=(256, 64), biasDim=64)
    # net = layersHDA.zRelu(net)

    # net = layersHDA.affine(net, weightShape=(64, 10), biasDim=10)
    # net = layersHDA.cTanh(net)

    # net = layersHDA.Phase(net)

    logits = CVNNLayer.affine(net, weightShape=(10, out_nums), biasDim=out_nums)
    # logits = layersHDA.normalize(logits)
    # logits = layersHDA.Phase(logits)
    logits = CVNNLayer.reshape_cp(logits, [-1, out_nums])

    return logits


def TDNN(x, out_nums, keep_prob, mu=0, sigma=0.1, networkType='Complex'):

    # if networkType is not 'Real':
    # x = layersHDA.typeClone(x, networkType)
    # x = layersHDA.normalize(x)

    conv1 = CVNNLayer.conv1d(x,
        weightShape=(5, 1, 8),
        biasDim=8,
        convStride=1,
        convPadding='VALID')
    conv1 = CVNNLayer.cTanh(conv1)
    conv2 = CVNNLayer.conv1d(conv1,
        weightShape=(5, 8, 4),
        biasDim=4,
        convStride=1,
        convPadding='VALID')
    conv2 = CVNNLayer.cTanh(conv2)
    conv3 = CVNNLayer.conv1d(conv2,
        weightShape=(5, 4, 2),
        biasDim=2,
        convStride=1,
        convPadding='VALID')
    conv3 = CVNNLayer.cTanh(conv3)
    conv4 = CVNNLayer.conv1d(conv3,
        weightShape=(5, 2, 2),
        biasDim=2,
        convStride=1,
        convPadding='VALID')
    conv4 = CVNNLayer.cTanh(conv4)
    conv5 = CVNNLayer.conv1d(conv4,
        weightShape=(5, 2, 1),
        biasDim=1,
        convStride=1,
        convPadding='VALID')

    out = tf.squeeze(tf.transpose(conv5, [0, 2, 1])) # 1,46,1 --> 1,1,46
    out = CVNNLayer.affine(out, weightShape=(46, 10), biasDim=10)
    out = CVNNLayer.cTanh(out)
    logits = CVNNLayer.affine(out, weightShape=(10, 1), biasDim=1)
    # logits = (pi/2)*CVNNLayer.cTanh(logits)

    # logits = CVNNLayer.affine(out, weightShape=(10, out_nums), biasDim=out_nums)
    # logits = CVNNLayer.reshape_cp(logits, [-1, out_nums])

    return logits



def task_1d(net, out_nums=1, keep_prob=None):
    '''
    # Input = 5x16. Output = 3x32.只对第二维度作用池化
    conv1 = CVNNLayer.conv1d(net, (3, 16, 32), 32, 1, 'SAME')
    conv1 = CVNNLayer.phase_amplitude(conv1)
    conv1 = CVNNLayer.maxpool(conv1, size=[2], strides=[2], padding='SAME')
    # Output = 2x32
    conv2 = CVNNLayer.conv1d(conv1, (3, 32, 32), 32, 1, 'SAME')
    conv2 = CVNNLayer.phase_amplitude(conv2)
    conv2 = CVNNLayer.maxpool(conv2, size=[2], strides=[2], padding='SAME')
    '''
    # Input_num = 36
    Input_num = 8
    fc0 = CVNNLayer.reshape_cp(net, [-1, Input_num])

    # Fully Connected. Input = 50x128. Output = 50x32. fc1.r.shape[-1]=32
    fc1 = CVNNLayer.affine(
        fc0, weightShape=(Input_num, 10), biasDim=10, kp=keep_prob)
    fc1 = CVNNLayer.cTanh(fc1)
    # fc1 = CVNNLayer.normalize(fc1)

    # Input = 50x32. Output = 50x10.
    fc2 = CVNNLayer.affine(
        fc1, weightShape=(10, out_nums), biasDim=out_nums, kp=keep_prob)
    # fc2 = CVNNLayer.cTanh(fc2)
    # fc2 = CVNNLayer.cos_sin(fc2)

    # logits = CVNNLayer.reshape_cp(fc2, (-1, out_nums*2))
    # fc2 = CVNNLayer.normalize(fc2)
    fc2 = CVNNLayer.Phase(fc2)  # 归一化幅度，虚部即正弦值
    
    '''
    # carlibrate fc2 = fc2/(0.5*pi)
    fc2 = CVNNLayer.affine(
        fc2, weightShape=(out_nums, 10), biasDim=10, kp=keep_prob)
    fc2 = CVNNLayer.cTanh(fc2)
    fc2 = CVNNLayer.affine(
        fc2, weightShape=(10, out_nums), biasDim=out_nums, kp=keep_prob)
    '''

    logits = CVNNLayer.reshape_cp(fc2, (-1, out_nums))

    return logits


def task_2d(net, out_nums=1, keep_prob=None):
    # Input = 5x16. Output = 3x32.只对第二维度作用池化
    # conv1 = layersHDA.conv1d(net, (3, 16, 32), 32, 1, 'SAME')
    # conv1 = layersHDA.cTanh(conv1)
    # conv1 = layersHDA.maxpool(conv1, size=[2], strides=[2], padding='SAME')
    # Output = 2x32
    # conv2 = layersHDA.conv1d(conv1, (3, 32, 32), 32, 1, 'SAME')
    # conv2 = layersHDA.cTanh(conv2)
    # conv2 = layersHDA.maxpool(conv2, size=[2], strides=[2], padding='SAME')

    Input_num = 72

    fc0 = CVNNLayer.reshape_cp(net, [-1, Input_num])

    # Fully Connected. Input = 50x128. Output = 50x32. fc1.r.shape[-1]=32
    fc1 = CVNNLayer.affine(
        fc0, weightShape=(Input_num, 10), biasDim=10, kp=keep_prob)
    fc1 = CVNNLayer.cTanh(fc1)
    # fc1 = layersHDA.normalize(fc1)

    # Input = 50x32. Output = 50x10.
    fc2 = CVNNLayer.affine(
        fc1, weightShape=(10, out_nums), biasDim=out_nums, kp=keep_prob)
    # fc2 = layersHDA.cTanh(fc2)
    # fc2 = layersHDA.normalize(fc2)
    fc2 = CVNNLayer.Phase(fc2)  # 归一化幅度，虚部即正弦值
    # logits = logits * pi / 2
    logits = CVNNLayer.reshape_cp(fc2, (-1, out_nums))

    return logits

