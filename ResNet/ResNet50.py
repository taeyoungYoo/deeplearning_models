'''
    Implementation of ResNet-50 (https://arxiv.org/abs/1801.04381)
'''

import tensorflow as tf


def identity_block(X, f, filters):
    '''
    Identity block
    :param X: input data
    :param f: int. shape of middle convnet's window
    :param filters: list[int], num of filters
    :return: Tensor type output
    '''
    F1, F2, F3 = filters
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)  # Default axis
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, s = 2):
    '''
    Conv block
    :param X: input data
    :param f: int. shape of middle convnet's window
    :param filters: list[int], num of filters
    :param s: int, stride
    :return: Tensor type output
    '''
    F1, F2, F3 = filters
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    ##### SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)

    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet50(input_shape = (224, 24, 3), num_classes = 1000):
    '''
    ResNet-50
    :param input_shape: tuple of int, shape of input image
    :param num_classes: int, num of classes
    :return: keras model instance
    '''
    X_input = tf.keras.layers.Input(input_shape)

    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = tf.keras.layers.AveragePooling2D((2, 2))(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(num_classes, activation='softmax')(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model