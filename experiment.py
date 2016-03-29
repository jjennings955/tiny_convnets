import glob
import os
from keras.layers import Convolution2D, Activation, AveragePooling2D, Flatten, Dense
from keras.regularizers import l2
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import scipy.misc

def build_model():
    model = Sequential()
    # Create a convolutional layer with 4 3x3 filters. Pad the output to be the same size as the input.
    model.add(Convolution2D(4, 3, 3, border_mode='same', input_shape=(1, 28, 28), W_regularizer=l2(0.0)))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D((14,14))) # Do rather aggressive pooling to limit the number of parameters.
    model.add(Flatten())
    model.add(Dense(3)) # 3 classes (X, O, Junk)
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def read_grayscale(dir, name):
    return scipy.misc.imread(os.path.join(dir, name), flatten=True)[np.newaxis, :, :]/255.0

def get_data(dir='./data', junk=False):
    X = read_grayscale(dir, 'X.png')
    O = read_grayscale(dir, 'O.png')

    # Create a dataset by simply duplicating the example images to create a class balance with the 'junk' class.
    # This seems to work better than weighing the samples.
    data = [X]*15 + [O]*15
    labels = [0]*15 + [1]*15

    if junk:
        junk_files = glob.glob(os.path.join(dir, 'J*.png'))
        junk_images =  [read_grayscale(data, fname) for fname in junk_files]
        for image in junk_images:
            data.append(image)
            labels.append(2)

    return np.array(data), np_utils.to_categorical(labels, 3)