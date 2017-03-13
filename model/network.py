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
from keras.layers import Dense, Dropout, Activation, Input, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from keras.layers import Convolution3D, MaxPooling3D

from src.config import config
from src.data_loader import *

def cnn_3d_net():
    model = Sequential()
    model.add(Convolution3D(nb_filter=config.FILTERS[0],
                            kernel_dim1=config.CONV_1[0],
                            kernel_dim2=config.CONV_1[1],
                            kernel_dim3=config.CONV_1[2],
                            input_shape=(1,config.FRAME_LEN,32,120),
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

def train(model, epoches):

    # load data
    filename_dict = file_dict()
    dataset_data, dataset_label = data_to_image(filename_dict)
    print(dataset_data.shape)
    X_train = dataset_data
    Y_train = np_utils.to_categorical(dataset_label, config.NUM_CLASSES)
    # ========Preprocessing===
    X_train = X_train.astype('float32')
    X_train -= np.mean(X_train)
    X_train /= np.max(X_train)

    tensorBoard = TensorBoard(log_dir=config.LOG_DIR, histogram_freq=10, write_graph=True)
    checkpointer = ModelCheckpoint(config.WEIGTH_PATH,verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_acc', patience=20)
    start_time = time.time()
    print(start_time)
    history = model.fit(X_train,Y_train,
                        batch_size=config.BATCH_SIZE,
                        nb_epoch=epoches,
                        validation_split=config.VAL_PERCENT,
                        shuffle=True,
                        callbacks=[tensorBoard,checkpointer,earlystopping])
    plot_curve(start_time, epoches, history)

def plot_curve(start_time, epoches, history):
    average_time_per_epoch = (time.time() - start_time) / epoches
    results = []
    results.append((history, average_time_per_epoch))

    modes = config.DEVICE
    #===========plot
    plt.style.use('ggplot')
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Train Loss')
    ax1.set_xlabel('Epochs')
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Loss')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xlabel('Epochs')
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title('Time')
    ax3.set_ylabel('Seconds')
    for mode, result in zip(modes, results):
        ax1.plot(result[0].epoch, result[0].history['val_acc'], label=mode)
        ax2.plot(result[0].epoch, result[0].history['val_loss'], label=mode)
    ax1.legend()
    ax2.legend()
    ax3.bar(np.arange(len(results)), [x[1] for x in results],
            tick_label=modes, align='center')
    plt.tight_layout()
    plt.savefig(config.CURVE_PATH, dpi=200)


def main():
    model = cnn_3d_net()
    train(model, epoches=1000)

if __name__ == '__main__':
    main()
