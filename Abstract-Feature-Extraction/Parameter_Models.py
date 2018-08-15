from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
import keras.backend as K
from keras import regularizers
from random import *
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

#Customized loss function of train on two parameters with available datapoint
def get_customLoss():
    def customLoss(y_true,y_pred):
        y_zeros = y_pred-y_pred
        y_true_temp = K.switch(tf.is_nan(y_true),y_zeros,y_true)
        y_pred_temp = K.switch(tf.is_nan(y_true),y_zeros,y_pred)
        #to find the number of available datapoints
        num = 200-tf.reduce_sum(tf.cast(tf.is_nan(y_true), tf.float32))
        return K.sum(K.square(y_pred_temp - y_true_temp),axis=None)/num
    return customLoss

#Han's customized model
def more_conv_multiple(input_dim, n_parameters):
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

    model.compile(loss='mse', optimizer='adam')

    return model

#Transfer Learning model using vgg19 and 'imagenet' weights
def vgg19_custom(n_parameters):

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))

    x = base_model.output
    # let's add a fully-connected layer
    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)

    x = Dense(32, kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.1)(x)

    predictions = Dense(n_parameters, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    #Freeze the conv-layer weights
    for layer in base_model.layers:
        layer.trainable = False

    #When training on multiple parameters, use the customized loss function
    if(n_parameters>1):
        print("Using custom loss function:")
        model.compile(loss=get_customLoss(),optimizer='adam')
    #When training on single parameter, use the built-in "mean-square-error" loss function
    elif(n_parameters==1):
        print("Using default mean-square-error loss function:")
        model.compile(loss='mse',optimizer='adam')

    return model

#Transfer Learning using ResNet model, not giving good result. See https://github.com/keras-team/keras/issues/9214 for potential problems with this approach.
def ResNet_custom(n_parameters):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

    x = base_model.output
    # let's add a fully-connected layer
    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)

    x = Dense(32, kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.1)(x)

    predictions = Dense(n_parameters, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[:-10]:
        layer.trainable = False
    # model.add(Dense(128, activation='relu'))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(32))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.1))
    #
    # model.add(Dense(n_parameters, activation='linear'))

    model.compile(loss=get_customLoss(),optimizer='adam')
    return model

# Han's Old Code for other models
# def original(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(input_dim, input_dim, 3)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(4, 4)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#
#     # model.add(Dense(1, activation='linear'))
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
#
# def gray(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(8, (9, 9), padding='same', input_shape=(input_dim, input_dim, 1)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(4, 4)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#
#     # model.add(Dense(1, activation='linear'))
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
#
# def gray2(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(16, (9, 9), padding='same', input_shape=(input_dim, input_dim, 1)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(4, 4)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
#
# def more_conv(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(8, (7, 7), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(4, 4)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
#
#
#
# def more_conv2(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(8, (7, 7), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
#
# def more_conv3(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(8, (7, 7), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
#
# def more_conv4(input_dim):
#     model = Sequential()
#     model.add(Convolution2D(8, (9, 9), padding='same', strides=(2, 2), input_shape=(input_dim, input_dim, 1)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(8, (7, 7), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(16, (5, 5), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(16, (3, 3), padding='same', strides=(2, 2)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.1))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Convolution2D(32, (3, 3), padding='same'))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2)))
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dropout(0.4))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.1))
#
#     model.add(Dense(1))
#
#     model.compile(loss='mse', optimizer='adam')
#
#     return model
