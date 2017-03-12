from __future__ import print_function
import os
import numpy as np
import cPickle
import h5py
import time

import matplotlib.pyplot as plt
from constants import COMBINE_FEATURE_DIR_TRAIN
from constants import COMBINE_FEATURE_DIR_VAL
from constants import FEATURE_DIM
from constants import FACE_MAX
from constants import FACE_MAX_2
from constants import CLIP_FRAME_NUMBER
from constants import LSTM_HIDDEN_STATE
from constants import OVERLAP_STEP
from constants import RNN_MODEL_JSON_PATH
from constants import COMBINE_FEATURE_DIR

def dataReadCrossval():

    totalFeatureTrain = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, FEATURE_DIM])
    totalGrTruTrain = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, 1])

    totalFeatureVal = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, FEATURE_DIM])
    totalGrTruVal = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, 1])

    h5FilenameListTrain = []
    h5FilenameListVal = []


    for (dirpath, dirnames, filenames) in os.walk('./data/specialvideo/video13'):
        for item in filenames:
            h5filePath = os.path.join(dirpath,item)
            print(h5filePath)
            h5FilenameListTrain.append(h5filePath)

    for (dirpath, dirnames, filenames) in os.walk(COMBINE_FEATURE_DIR_VAL):
        for item in filenames:
            h5filePath = os.path.join(dirpath,item)
            print(h5filePath)
            h5FilenameListVal.append(h5filePath)

    # for (dirpath, dirnames, filenames) in os.walk(COMBINE_FEATURE_DIR_TRAIN):
    #     for filename in filenames:
    #         h5FilenameListTrain.append(filename)
    #
    # for (dirpath, dirnames, filenames) in os.walk(COMBINE_FEATURE_DIR_VAL):
    #     for filename in filenames:
    #         h5FilenameListVal.append(filename)

    for item in h5FilenameListTrain:
        print(item)
        # path = os.path.join(COMBINE_FEATURE_DIR_TRAIN,item)
        with h5py.File(item, 'r') as f:
            clipFeatureMultiFaceTemp = np.array([f['data']])
            clipFeatureMultiFace = np.vstack([clipFeatureMultiFaceTemp[0],clipFeatureMultiFaceTemp[1],clipFeatureMultiFaceTemp[2],clipFeatureMultiFaceTemp[3],clipFeatureMultiFaceTemp[7],clipFeatureMultiFaceTemp[8]])
            clipGrTruMultiFaceTemp = np.array([f['grtru']])
            clipGrTruMultiFace = np.vstack([clipGrTruMultiFaceTemp[0],clipGrTruMultiFaceTemp[1],clipGrTruMultiFaceTemp[2],clipGrTruMultiFaceTemp[3],clipGrTruMultiFaceTemp[7],clipGrTruMultiFaceTemp[8]])

            f.close()
        totalFeatureTrain = np.vstack([totalFeatureTrain, clipFeatureMultiFace])
        totalGrTruTrain = np.vstack([totalGrTruTrain, clipGrTruMultiFace])

    # for item in h5FilenameListVal:
    #     # print(item)
    #     # path = os.path.join(COMBINE_FEATURE_DIR_VAL,item)
    #     with h5py.File(item, 'r') as f:
    #         clipFeatureMultiFace = np.array([f['data']])
    #         clipGrTruMultiFace = np.array([f['grtru']])
    #         f.close()
    #     totalFeatureVal = np.vstack([totalFeatureVal, clipFeatureMultiFace])
    #     totalGrTruVal = np.vstack([totalGrTruVal, clipGrTruMultiFace])

    totalFeatureTrain = totalFeatureTrain[1:]
    totalGrTruTrain = totalGrTruTrain[1:]

    # totalFeatureVal = totalFeatureVal[1:]
    # totalGrTruVal = totalGrTruVal[1:]

    print(totalFeatureTrain.shape, totalGrTruTrain.shape)
    # print(totalFeatureVal.shape, totalGrTruVal.shape)

    return totalFeatureTrain, totalGrTruTrain #, totalFeatureVal, totalGrTruVal

def saveCrossRNNData():
    totalFeatureTrain, totalGrTruTrain = dataReadCrossval()
    trainh5FilePath = os.path.join('./data/specialvideo','video13_trainTotalRNN_overlap{}.h5'.format(OVERLAP_STEP))
    print(trainh5FilePath)
    with h5py.File(trainh5FilePath, 'w') as f:
        f['data'] = totalFeatureTrain
        f['grtru'] = totalGrTruTrain
        f.close()

    # valh5FilePath = os.path.join('./data/specialvideo','testTotalRNN_overlap{}.h5'.format(OVERLAP_STEP))
    # print(valh5FilePath)
    # with h5py.File(valh5FilePath, 'w') as f:
    #     f['data'] = totalFeatureVal
    #     f['grtru'] = totalGrTruVal
    #     f.close()
if __name__ == '__main__':
    saveCrossRNNData()
