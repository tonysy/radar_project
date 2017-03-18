import numpy as np
from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 32
config.FRAME_LEN = 12

config.FILTERS = [25, 25]
config.CONV_1 = [5, 5, 5]
config.POOL_1 = [2, 2, 2]
config.CONV_2 = [3, 3, 3]
config.POOL_2 = [2, 2, 2]

config.NUM_CLASSES = 4
config.MODEL_JSON_PATH = './network.json'
config.LOG_DIR = './logs/'
config.WEIGTH_PATH = './model/cnn_3d_weigth_rmsprop_0.h5'

config.VAL_PERCENT = 0.4
config.DEVICE = ['GPU']

config.CURVE_PATH = './model/loss_curve.png'
config.DATASET_PATH = './data'


config.DATA_MEAN = 0.54246
config.DATA_MAX = 1130.89