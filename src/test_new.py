from __future__ import print_function
import sys
sys.path.insert(0, '/home/syzhang/Documents/radar_project/')
import numpy as np
from net.testor import Gesture_Testor
from src.data_loader import *
from src.config import config
from src.get_intermediate_layer import get_intermediate_layer
from net.train import cnn_3d_net
def load_data():
    filename_dict = file_dict(config.DATASET_PATH)
    dataset_data, dataset_label = data_to_image(filename_dict)

    print(dataset_data.shape)
    X_train = dataset_data
    X_train = X_train.astype('float32')

    X_test = X_train[0]
    X_test = np.array([X_test])
    print(X_test.shape)
    return X_test


test_data = load_data()
test_data -= config.DATA_MEAN
test_data /= config.DATA_MAX

gesture_testor = Gesture_Testor()
gesture_testor.test(test_data)

# get intermediate layers
layer_name = "convolution3d_1"
output = get_intermediate_layer(layer_name, test_data)
print(output)
