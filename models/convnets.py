from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras import backend as K

#Todo : make it more customizable.

def build_cnn(input_shape, nb_classes, kernel_size, pool_size, nb_filters, dropout=0.5,activation='relu'):
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation(activation))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=pool_size))
    if dropout > 0:
        model.add(Dropout(dropout*0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    if dropout > 0 :
        model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model
