from __future__ import print_function
import numpy as np
from net.testor import Gesture_Testor
from src.data_loader import *
from src.config import config
from keras.utils import np_utils

def load_data():
    filename_dict = file_dict()
    dataset_data, dataset_label = data_to_image(filename_dict)
    print(dataset_data.shape)
    X_train = dataset_data
    X_train = X_train.astype('float32')
    X_test = X_train[0]
    X_test = np.array([X_test])
    print(X_test.shape)
    return X_test

def evaluate_data():
    filename_dict_train = file_dict('./data_split/train')
    dataset_data_train, dataset_label_train = data_to_image(filename_dict_train)
    print(dataset_data_train.shape)
    X_train = dataset_data_train
    Y_train = np_utils.to_categorical(dataset_label_train, config.NUM_CLASSES)
    X_train = X_train.astype('float32')

    filename_dict_test = file_dict('./data_split/test')
    dataset_data_test, dataset_label_test = data_to_image(filename_dict_test)
    print(dataset_data_test.shape)
    X_test = dataset_data_test
    Y_test = np_utils.to_categorical(dataset_label_test, config.NUM_CLASSES)
    X_test = X_test.astype('float32')

    # =========Calculate Mean
    X_total = np.vstack([X_train, X_test])
    X_MEAN = np.mean(X_total)
    print(X_MEAN)
    X_test -= X_MEAN
    print(np.max(X_test))
    print(np.max(X_train))
    print(np.max(X_total))
    X_test /= np.max(X_total)

    return  X_test, Y_test

# test_data = load_data()
X_test, Y_test = evaluate_data()

loss_and_metrics = Gesture_Testor().model.evaluate(X_test,Y_test)
print(loss_and_metrics)
print(type(loss_and_metrics))