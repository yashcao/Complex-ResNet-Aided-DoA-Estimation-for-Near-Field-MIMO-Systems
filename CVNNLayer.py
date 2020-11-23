# import numpy as np
from numpy import pi
import tensorflow as tf


class TFHDA:
    '''
    Class for a tensorflow high dimensional algebraic object
        - Designed to interface with
        - Current implementation is only for 'Complex' and 'SplitComplex'
    '''
    def __init__(self, real, imag, HDAtype='Complex'):
        self.r = real
        self.i = imag
        self.HDAtype = HDAtype


def typeClone(x, HDAtype):
    '''
    Clones input 'x' to be correct high-dimensional algebra type
    '''
    # xout = TFHDA(x, tf.zeros_like(x), HDAtype=HDAtype)
    # x_real = x.real.astype('f')
    # x_imag = x.imag.astype('f')
    x_real = tf.real(x)
    x_imag = tf.imag(x)
    xout = TFHDA(x_real, x_imag, HDAtype=HDAtype)
    return xout


def conv1d(x,
           weightShape=None,
           biasDim=None,
           convStride=None,
           convPadding=None,
           mu=0,
           sigma=0.1):

    if type(x) is TFHDA:
        # conv_W = tf.Variable(tf.random_uniform(
        #     shape=weightShape, minval=-pi, maxval=pi, dtype=tf.float32))
        # conv_W_r = tf.Variable(tf.cos(conv_W))
        # conv_W_i = tf.Variable(tf.sin(conv_W))
        conv_W_r = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_W_i = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_b_r = tf.Variable(tf.zeros(biasDim))
        conv_b_i = tf.Variable(tf.zeros(biasDim))

        if x.HDAtype == 'Complex':
            convOutr = tf.nn.conv1d(
                x.r, conv_W_r, stride=convStride, padding=convPadding) - tf.nn.conv1d(
                x.i, conv_W_i, stride=convStride,
                padding=convPadding) + conv_b_r

            convOuti = (tf.nn.conv1d(
                x.i, conv_W_r, stride=convStride, padding=convPadding) + tf.nn.conv1d(
                x.r, conv_W_i, stride=convStride,
                padding=convPadding) + conv_b_i)

            convOut = TFHDA(convOutr, convOuti)

        elif x.HDAtype == 'SplitComplex':
            convOutr = tf.nn.conv1d(
                x.r, conv_W_r, stride=convStride, padding=convPadding)
            +tf.nn.conv1d(
                x.i, conv_W_i, stride=convStride,
                padding=convPadding) + conv_b_r
            convOuti = tf.nn.conv1d(
                x.i, conv_W_r, stride=convStride, padding=convPadding)
            +tf.nn.conv1d(
                x.r, conv_W_i, stride=convStride,
                padding=convPadding) + conv_b_i
            convOut = TFHDA(convOutr, convOuti)

    else:
        conv_W = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_b = tf.Variable(tf.zeros(biasDim))
        convOut = tf.nn.conv1d(
            x, conv_W, stride=convStride, padding=convPadding) + conv_b

    return convOut


def conv2d(x, weightShape=None, biasDim=None, convStride=None, convPadding=None, mu=0, sigma=0.1):
    '''
    2D convolutional layer overloaded for complex arithmetic, implements via weight-sharing and
        works as wrapper file for tf.nn.conv2d()
    Inputs:
        x - input array
        weightShape - tuple with shape
        biasDim - size of bias, should match last dimension of weightShape
        convStride - array of strides for the convolution
        convPadding - padding time for convolution
        mu - mean for weight initialization
        sigma - std dev for weight initialization
    Output:
        convOut - convolutional layer output
    '''

    if type(x) is TFHDA:
        conv_W_r = tf.Variable(tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_W_i = tf.Variable(tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_b_r = tf.Variable(tf.zeros(biasDim))
        conv_b_i = tf.Variable(tf.zeros(biasDim))

        convOutr = (tf.nn.conv2d(x.r, conv_W_r, strides=convStride, padding=convPadding) - tf.nn.conv2d(x.i, conv_W_i, strides=convStride, padding=convPadding) + conv_b_r)

        convOuti = (tf.nn.conv2d(x.i, conv_W_r, strides=convStride, padding=convPadding) + tf.nn.conv2d(x.r, conv_W_i, strides=convStride, padding=convPadding) + conv_b_i)

        convOut = TFHDA(convOutr, convOuti)

    else:
        conv_W = tf.Variable(tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_b = tf.Variable(tf.zeros(biasDim))
        convOut = tf.nn.conv2d(x, conv_W, strides=convStride, padding=convPadding) + conv_b

    return convOut


def affine(x, weightShape=None, biasDim=None, mu=0, sigma=0.01, kp=None):
    if type(x) is TFHDA:
        # fc_W = tf.Variable(tf.random_uniform(
        #     shape=weightShape, minval=-pi, maxval=pi, dtype=tf.float32))
        # fc_W_r = tf.Variable(tf.cos(fc_W))
        # fc_W_i = tf.Variable(tf.sin(fc_W))
        fc_W_r = tf.Variable(tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        fc_W_i = tf.Variable(tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        fc_b_r = tf.Variable(tf.zeros(biasDim))
        fc_b_i = tf.Variable(tf.zeros(biasDim))

        affineOutr = tf.matmul(x.r, fc_W_r) - tf.matmul(x.i, fc_W_i) + fc_b_r
        affineOuti = tf.matmul(x.r, fc_W_i) + tf.matmul(x.i, fc_W_r) + fc_b_i
        # dropout
        # shape = affineOutr.get_shape().as_list()[-1]
        # mask = tf.nn.dropout(tf.ones(shape), keep_prob=kp)
        # affineOutr = tf.multiply(affineOutr, mask)
        # affineOuti = tf.multiply(affineOuti, mask)

        affineOut = TFHDA(affineOutr, affineOuti)

    else:
        fc_W = tf.Variable(tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        fc_b = tf.Variable(tf.zeros(biasDim))
        affineOut = tf.matmul(x, fc_W) + fc_b
        # affineOut = tf.nn.dropout(affineOut, keep_prob=kp)

    return affineOut


def conv1d_w(x, weightShape=None, biasDim=None, convStride=None, convPadding=None, mu=0, sigma=0.01):

    if type(x) is TFHDA:
        fc_W = tf.random_uniform(shape=weightShape, minval=(-1 * pi), maxval=pi)
        # fc_W = tf.exp(1j * fc_W)

        conv_W_r = tf.Variable(tf.cos(fc_W))
        conv_W_i = tf.Variable(tf.sin(fc_W))
        '''
        conv_W_r = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_W_i = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        '''
        conv_b_r = tf.Variable(tf.zeros(biasDim))
        conv_b_i = tf.Variable(tf.zeros(biasDim))

        if x.HDAtype == 'Complex':
            convOutr = tf.nn.conv1d(
                x.r, conv_W_r, stride=convStride,
                padding=convPadding) - tf.nn.conv1d(
                    x.i, conv_W_i, stride=convStride,
                    padding=convPadding) + conv_b_r

            convOuti = (tf.nn.conv1d(
                x.i, conv_W_r, stride=convStride,
                padding=convPadding) + tf.nn.conv1d(
                    x.r, conv_W_i, stride=convStride, padding=convPadding) +
                        conv_b_i)

            convOut = TFHDA(convOutr, convOuti)

        elif x.HDAtype == 'SplitComplex':
            convOutr = tf.nn.conv1d(
                x.r, conv_W_r, stride=convStride, padding=convPadding)
            +tf.nn.conv1d(
                x.i, conv_W_i, stride=convStride,
                padding=convPadding) + conv_b_r
            convOuti = tf.nn.conv1d(
                x.i, conv_W_r, stride=convStride, padding=convPadding)
            +tf.nn.conv1d(
                x.r, conv_W_i, stride=convStride,
                padding=convPadding) + conv_b_i
            convOut = TFHDA(convOutr, convOuti)

    else:
        conv_W = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        conv_b = tf.Variable(tf.zeros(biasDim))
        convOut = tf.nn.conv1d(
            x, conv_W, stride=convStride, padding=convPadding) + conv_b

    return convOut


def affine_w(x, weightShape=None, biasDim=None, mu=0, sigma=0.01):
    if type(x) is TFHDA:
        fc_W = tf.random_uniform(shape=weightShape, minval=(-1 * pi), maxval=pi)
        # fc_W = tf.exp(1j * fc_W)

        fc_W_r = tf.Variable(tf.cos(fc_W))
        fc_W_i = tf.Variable(tf.sin(fc_W))
        # fc_W_r = tf.Variable(
        #     tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        # fc_W_i = tf.Variable(
        #     tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        fc_b_r = tf.Variable(tf.zeros(biasDim))
        fc_b_i = tf.Variable(tf.zeros(biasDim))

        affineOutr = tf.matmul(x.r, fc_W_r) - tf.matmul(x.i, fc_W_i) + fc_b_r
        affineOuti = tf.matmul(x.r, fc_W_i) + tf.matmul(x.i, fc_W_r) + fc_b_i

        affineOut = TFHDA(affineOutr, affineOuti)

    else:
        fc_W = tf.Variable(
            tf.truncated_normal(shape=weightShape, mean=mu, stddev=sigma))
        fc_b = tf.Variable(tf.zeros(biasDim))
        affineOut = tf.matmul(x, fc_W) + fc_b

    return affineOut


def avgpool(x, size=None, strides=None, padding=None):
    '''
    Performs average pooling. If input x is
    '''

    if type(x) is TFHDA:
        # mags = magnitude(x)
        out_r = tf.nn.pool(
            input=x.r,
            window_shape=size,
            strides=strides,
            pooling_type="AVG",
            padding=padding)
        out_i = tf.nn.pool(
            input=x.i,
            window_shape=size,
            strides=strides,
            pooling_type="AVG",
            padding=padding)
        out = TFHDA(out_r, out_i)

    else:
        out = tf.nn.pool(
            input=x,
            window_shape=size,
            strides=strides,
            pooling_type="AVG",
            padding=padding)

    return out


def maxpool(x, size, strides, padding):
    if type(x) is TFHDA:
        # mags = magnitude(x)
        # window_shape=[2]
        out_r = tf.nn.pool(
            input=x.r,
            window_shape=size,
            strides=strides,
            pooling_type="MAX",
            padding=padding)
        out_i = tf.nn.pool(
            input=x.i,
            window_shape=size,
            strides=strides,
            pooling_type="MAX",
            padding=padding)
        out = TFHDA(out_r, out_i)

    else:
        out = tf.nn.pool(
            input=x, window_shape=size,
            strides=strides, pooling_type="MAX", padding=padding)

    return out


def pool_2D(x, size, strides, padding, pooling_type='AVG'):
    if pooling_type == 'MAX':
        if type(x) is TFHDA:
            out_r = tf.nn.max_pool(
                value=x.r, ksize=size, strides=strides, padding=padding)
            out_i = tf.nn.max_pool(
                value=x.i, ksize=size, strides=strides, padding=padding)
            out = TFHDA(out_r, out_i)

        else:
            out = tf.nn.max_pool(value=x, ksize=size, strides=strides, padding=padding)

        return out

    elif pooling_type == 'AVG':
        if type(x) is TFHDA:
            out_r = tf.nn.avg_pool(value=x.r, ksize=size, strides=strides, padding=padding)
            out_i = tf.nn.avg_pool(value=x.i, ksize=size, strides=strides, padding=padding)
            out = TFHDA(out_r, out_i)

        else:
            out = tf.nn.avg_pool(
                value=x, ksize=size, strides=strides, padding=padding)

        return out


def relu(x):
    '''
    Implements ReLU function as defined in the paper
    '''

    if type(x) is TFHDA:
        mask = tf.cast(tf.greater_equal(x.r, tf.zeros_like(x.r)), tf.float32)
        out = TFHDA(tf.multiply(mask, x.r), tf.multiply(mask, x.i))
    else:
        out = tf.nn.relu(x)

    return out


def zRelu(x):
    '''
    Implements ReLU function as defined in the paper
    '''

    if type(x) is TFHDA:
        phase = tf.atan(x.i/x.r)
        mask1 = tf.cast(tf.greater_equal(phase, tf.zeros_like(phase)), tf.float32)
        mask2 = tf.cast(tf.greater_equal(tf.ones_like(phase)*(0.5*pi), phase), tf.float32)
        mask = tf.multiply(mask1, mask2)
        out = TFHDA(tf.multiply(mask, x.r), tf.multiply(mask, x.i))
    else:
        out = tf.nn.relu(x)

    return out


def fRelu(x):

    if type(x) is TFHDA:
        '''
        phase = tf.atan(x.i / x.r)
        mask1 = tf.cast(
            tf.greater_equal(phase,
                             tf.ones_like(phase) * (-0.5 * pi)), tf.float32)
        mask2 = tf.cast(
            tf.greater_equal(tf.ones_like(phase) * (0.5 * pi), phase),
            tf.float32)
        '''
        x.r = tf.cast(tf.abs(x.r), tf.float32)
        # mask = tf.multiply(mask1, mask2)
        out = normalize(TFHDA(x.r, x.i))
    else:
        out = tf.nn.relu(x)

    return out


def cSigmoid(x):
    '''
    Implements ReLU function as defined in the paper
    '''

    if type(x) is TFHDA:
        x_real = tf.nn.sigmoid(x.r)
        x_imag = tf.nn.sigmoid(x.i)
        out = TFHDA(x_real, x_imag)
    else:
        out = tf.nn.sigmoid(x)

    return out


def cTanh(x):
    '''
    Implements ReLU function as defined in the paper
    '''

    if type(x) is TFHDA:
        x_real = tf.nn.tanh(x.r)
        x_imag = tf.nn.tanh(x.i)
        out = TFHDA(x_real, x_imag)
    else:
        out = tf.nn.tanh(x)

    return out


def flatten(x):
    '''
    Flattens input array along each dimension if input is high-dimensional algebraic valued
    '''

    if type(x) is TFHDA:
        out = TFHDA(tf.squeeze(x.r), tf.squeeze(x.i))
    else:
        out = tf.squeeze(x)

    return out


def reshape_cp(x, tshape):
    '''
    Flattens input array along each dimension if input is high-dimensional algebraic valued
    '''

    if type(x) is TFHDA:
        out = TFHDA(tf.reshape(x.r, tshape), tf.reshape(x.i, tshape))
    else:
        out = tf.reshape(x, tshape)

    return out


def magnitude(x):
    '''
    Compute magnitude of input if input is HDA object.
        - If input is real-valued, pass through
    '''

    if type(x) is TFHDA:
        out = tf.sqrt(tf.square(x.r) + tf.square(x.i))
    else:
        out = x

    return out


def Phase(x):
    '''
    Compute magnitude of input if input is HDA object.
        - If input is real-valued, pass through
    '''

    if type(x) is TFHDA:
        # phase = tf.angle(c)
        # phase = tf.atan2(x.i, x.r)
        # x.r = tf.maximum(abs(x.r), 1e-15)

        phase = tf.div(x.i, tf.maximum(abs(x.r), 1e-15))
        # phase = tf.div(x.i, abs(x.r)+1e-15)
        phase = tf.atan(phase)

        # phase = x.i
        # phase = tf.concat([x.r, x.i], axis=1)  # shape(batch, 4)
    else:
        phase = 0

    return phase


def Ctan(x):
    '''
    Compute magnitude of input if input is HDA object.
        - If input is real-valued, pass through
    '''

    if type(x) is TFHDA:
        k = tf.square(x.r)
        d_real = x.r * k
        d_imag = x.i * k
        x = TFHDA(d_real, d_imag, 'Complex')
        # c = tf.complex(x.r, x.i)
        # phase = tf.angle(c)
        # phase = tf.atan2(x.i, x.r)
        phase = tf.div(x.i, x.r)
    else:
        phase = 0

    return phase

# def batch_norm():
#     batch_norm = tf.nn.batch_norm_with_global_normalization(
#         conv, mean, var, beta, gamma, 0.001,
#         scale_after_normalization=True)

#     BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)

#     out = relu(BN_out)
#     return out


def add_Complex(x, y):
    if type(x) is TFHDA:
        # z_r = x.r + y.r
        # z_i = x.i + y.i
        z_r = tf.add(x.r, y.r)
        z_i = tf.add(x.i, y.i)
        out = TFHDA(z_r, z_i, 'Complex')
    else:
        out = tf.add(x, y)
    return out


def normalize(data):
    data_m = tf.sqrt(tf.square(data.r) + tf.square(data.i))
    '''
    dmax = tf.reduce_max(data_m, 1)  # 1 代表按列取最大值
    dmin = tf.reduce_min(data_m, 1)
    data.r = (data.r - dmin) / (dmax - dmin)
    data.i = (data.i - dmin) / (dmax - dmin)
    '''
    # data = data.r + 1j*data.i
    # data = data / data_m
    d_real = data.r / data_m
    d_imag = data.i / data_m
    data = TFHDA(d_real, d_imag, 'Complex')

    return data


def phase_amplitude(x):
    '''
    Implements ReLU function as defined in the paper
    '''

    if type(x) is TFHDA:
        data_m = tf.sqrt(tf.square(x.r) + tf.square(x.i))
        x_n = normalize(x)
        x_real = x_n.r * tf.nn.tanh(data_m)
        x_imag = x_n.i * tf.nn.tanh(data_m)
        out = TFHDA(x_real, x_imag)
    else:
        out = tf.nn.tanh(x)

    return out


def finetune(data):
    data_m = tf.square(data.r) + tf.square(data.i)
    '''
    dmax = tf.reduce_max(data_m, 1)  # 1 代表按列取最大值
    dmin = tf.reduce_min(data_m, 1)
    data.r = (data.r - dmin) / (dmax - dmin)
    data.i = (data.i - dmin) / (dmax - dmin)
    '''
    # data = data.r + 1j*data.i
    # data = data / data_m
    # k = tf.div(data_m, tf.square(data.r))
    d_real = data.r * data_m
    d_imag = data.i * data_m
    data = TFHDA(d_real, d_imag, 'Complex')

    return data


def cos_sin(data):
    data_m = tf.sqrt(tf.square(data.r) + tf.square(data.i))

    # data = data.r + 1j*data.i
    # data = data / data_m
    d_real = data.r / data_m  # batch * 1
    d_imag = data.i / data_m
    # data = TFHDA(d_real, d_imag, 'Complex')
    data = tf.concat([d_real, d_imag], 0) # batch * 2

    return data