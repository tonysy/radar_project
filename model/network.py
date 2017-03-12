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
from keras.layers import Dense, Dropout, Activation, Input, TimeDistributed, Bidirectional
from keras.layers import LSTM, BatchNormalization, merge
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l2, activity_l2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import model_from_json

def cnn_3d_net():
