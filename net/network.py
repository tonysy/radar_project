from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import cPickle
import h5py
import time
import sys
import matplotlib.pyplot as plt
np.random.seed(1337)

from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.layers import Convolution3D, MaxPooling3D

from src.config import config

def cnn_3d_net():
    model = Sequential()
    model.add(Convolution3D(nb_filter=config.FILTERS[0],
                            kernel_dim1=config.CONV_1[0],
                            kernel_dim2=config.CONV_1[1],
                            kernel_dim3=config.CONV_1[2],
                            input_shape=(1,14,32,120),
                            activation='relu',
                            dim_ordering="th"))
    print("Load Conv1")
    model.add(MaxPooling3D(pool_size=(config.POOL_1[0],config.POOL_1[1],config.POOL_1[2]),dim_ordering="th"))
    print("Load pool1")
    model.add(Convolution3D(nb_filter=config.FILTERS[1],
                            kernel_dim1=config.CONV_2[0],
                            kernel_dim2=config.CONV_2[1],
                            kernel_dim3=config.CONV_2[2],
                            activation='relu',
                            dim_ordering="th"))
    print("Load Conv2")
    model.add(MaxPooling3D(pool_size=(config.POOL_2[0],config.POOL_2[1],config.POOL_2[2]), dim_ordering="th"))
    print("Load Conv2")
    model.add(Flatten())
    model.add(Dense(1024, init='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, init='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.NUM_CLASSES, init='normal'))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

    # save network architecture to json file
    json_string = model.to_json()
    open(config.MODEL_JSON_PATH,'w').write(json_string)
    print(model.summary())
    return model

def train():
    results = []
    tensorBoard = TensorBoard(log_dir=config.LOG_DIR, histogram_freq=10, write_graph=True)
    checkpointer = ModelCheckpoint(config.WEIGTH_PATH,verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    start_time = time.time()
    history = model.fit(X_train,Y_train , batch_size=batch_size, nb_epoch=epoches ,validation_split = validation_split,shuffle = True, callbacks=[tensorBoard,checkpointer,earlystopping])
