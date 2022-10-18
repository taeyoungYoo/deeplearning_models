'''
    Implementation of MobileNetV2 (https://www.sciencedirect.com/science/article/abs/pii/S0893608020304470)
'''

import tensorflow as tf


def expansion_layer(x, t, c_i):
    '''
    Expansion layer
    :param x: Tensor, input data
    :param t: int, expansion factor
    :param c_i: int, num of input blocks channel
    :return: Tensor, expanded layer
    '''
    return None


def depthwise_layer(x, s):
    '''
    Depthwise convolution layer
    :param x: Tensor, input data
    :param s: int, stride. Could be 1 or 2
    :return: Tensor, depthwised layer
    '''
    return None


def projection_layer(x, c_o):
    '''
    Pointwise convolution layer
    :param x: Tensor, input data
    :param c_o: intk num of output blocks channel
    :return: Tensor, end of block
    '''
    return None


def bottleneck_block(x, t, c, s):
    '''
    baseline of inverted residual block
    :param x: Tensor, input data
    :param t: int, expansion factor
    :param c: int, same as num of output channel
    :param s: int, stride. Could be 1 or 2
    :return: Tensor
    '''
    return None


def inverted_residual_block(x, t, c, n, s):
    '''
    Repeating the bottleneck block
    :param x: Tensor, input data
    :param t: int, expansion factor
    :param c: int, num of output channel
    :param n: int, repeated n times
    :param s: int, stride
    :return: Tensor
    '''
    return None


def QFMobileNetV2(input_shape, num_classes=1000):
    '''
    MobilNetV2
    :param input_shape: tuple consist of 3 integers
    :param num_classes: int. num of labels
    :return: qf_mobilenetv2 model
    '''
    return None