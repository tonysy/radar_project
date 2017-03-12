import numpy as np
from easydict import EasyDict as edict

config = edict()

config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.BATCH_SIZE = 32
config.FRAME_LEN = 12
config.
