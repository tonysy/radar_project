from __future__ import print_function
import numpy as np
from net.testor import Gesture_Testor
from src.data_loader import *

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

test_data = load_data()
label = Gesture_Testor().test(test_data)
