'''
    Implementation of MobileNetV2 (https://arxiv.org/abs/1801.04381)
    Baseline for project: MobileNetV2 for Quantization
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
    num_filter = t * c_i
    x = tf.keras.layers.Conv2D(filters=num_filter, kernel_size=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    return x


def depthwise_layer(x, s):
    '''
    Depthwise convolution layer
    :param x: Tensor, input data
    :param s: int, stride. Could be 1 or 2
    :return: Tensor, depthwised layer
    '''
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(s, s), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    return x


def projection_layer(x, c_o):
    '''
    Pointwise convolution layer
    :param x: Tensor, input data
    :param c_o: intk num of output blocks channel
    :return: Tensor, end of block
    '''
    num_filter = c_o
    x = tf.keras.layers.Conv2D(num_filter, kernel_size=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def bottleneck_block(x, t, c, s):
    '''
    baseline of inverted residual block
    :param x: Tensor, input data
    :param t: int, expansion factor
    :param c: int, same as num of output channel
    :param s: int, stride. Could be 1 or 2
    :return: Tensor
    '''
    assert s in [1, 2]
    c_i = x.shape[-1]
    c_o = c
    inp_block = expansion_layer(x, t, c_i)
    inp_block = depthwise_layer(inp_block, s)
    inp_block = projection_layer(inp_block, c_o)
    if x.shape == inp_block.shape:
        inp_block = tf.keras.layers.add([x + inp_block])
    return inp_block


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
    for i in range(n):
        x = bottleneck_block(x, t, c, s)
    return x


def MobileNetV2(input_shape, num_classes=1000):
    '''
    MobilNetV2
    :param input_shape: tuple consist of 3 integers
    :param num_classes: int. num of labels
    :return: mobilenetv2 model
    '''
    input_ = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2)(input_)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)

    x = inverted_residual_block(x, 1, 16, 1, 1)
    x = inverted_residual_block(x, 6, 24, 2, 2)
    x = inverted_residual_block(x, 6, 32, 3, 2)
    x = inverted_residual_block(x, 6, 64, 4, 2)
    x = inverted_residual_block(x, 6, 96, 3, 1)
    x = inverted_residual_block(x, 6, 160, 3, 2)
    x = inverted_residual_block(x, 6, 320, 1, 1)

    x = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output_ = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(input_, output_)
    return model