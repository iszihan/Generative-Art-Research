from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
import keras.backend as K
from random import *
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

def original(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(input_dim, input_dim, 3)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    # model.add(Dense(1, activation='linear'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def gray(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    # model.add(Dense(1, activation='linear'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def gray2(input_dim):
    model = Sequential()
    model.add(Convolution2D(16, (9, 9), padding='same', input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def get_customLoss():
    def customLoss(y_true,y_pred):
        y_zeros = y_pred-y_pred
        y_true_temp = K.switch(tf.is_nan(y_true),y_zeros,y_true)
        y_pred_temp = K.switch(tf.is_nan(y_true),y_zeros,y_pred)
        #to find the number of available datapoints
        num = 600-tf.reduce_sum(tf.cast(tf.is_nan(y_true), tf.float32))
        return K.sum(K.square(y_pred_temp - y_true_temp),axis=None)/num
    return customLoss

def more_conv_multiple(input_dim, n_parameters):

    # def get_customLoss():
    #     def customLoss(y_true,y_pred):
    #         y_zeros = y_pred-y_pred
    #         y_true_temp = K.switch(tf.is_nan(y_true),y_zeros,y_true)
    #         y_pred_temp = K.switch(tf.is_nan(y_true),y_zeros,y_pred)
    #         #to find the number of available datapoints
    #         num = 600-tf.reduce_sum(tf.cast(tf.is_nan(y_true), tf.float32))
    #         return K.sum(K.square(y_pred_temp - y_true_temp),axis=None)/num
    #     return customLoss

    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))

    model.add(Dense(n_parameters, activation='linear'))

    model.compile(loss=get_customLoss(),optimizer='adam')
    return model

def more_conv2(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv3(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def more_conv4(input_dim):
    model = Sequential()
    model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(8, (7, 7), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model
