from __future__ import print_function
import numpy as np
import h5py
import time
import sys

from keras.models import model_from_json
from keras.optimizers import RMSprop
from src.config import config
from src.data_loader import *


class Gesture_Testor(object):
    """docstring for Gesture_Testor."""
    def __init__(self):
        super(Gesture_Testor, self).__init__()
        self.optimizer = RMSprop(lr=0.0005)
        self.label = -1
    def load_model(self):
        self.model = model_from_json(open(config.MODEL_JSON_PATH).read())
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['mse', 'accuracy'])
        self.model.load_weights(config.WEIGTH_PATH)
    def test(self, X_test):
        self.load_model()
        self.label = self.model.predict_classes(X_test, batch_size=1,verbose=1)
        print(self.label)
