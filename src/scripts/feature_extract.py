from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import struct
import h5py
import time
from constants import GPU_ID
from constants import CAFFE_ROOT_DIR
from constants import MEAN_FILE_PATH
from constants import TRAIN_FILELIST_CNN
from constants import VAL_FILELIST_CNN
from constants import DEPLOY_TXT_PATH
from constants import INPUT_DIR
from constants import MODEL_FILE_PATH
from constants import FEATURE_LAYER
from constants import FEATURE_SINGLE_DIR_TRAIN
from constants import FEATURE_SINGLE_DIR_VAL
from constants import FEATURE_SINGLE_DIR
from constants import TOTAL_FILELIST_CNN

def initilize():
    print('initilize ... ')

    sys.path.insert(0, CAFFE_ROOT_DIR + 'python')
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    net = caffe.Net(DEPLOY_TXT_PATH, MODEL_FILE_PATH,caffe.TEST)
    return net

def featureExtract(fileList, net, featureSavePathBase):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(MEAN_FILE_PATH).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    net.blobs['data'].reshape(1,3,227,227)

    for idx in range(len(fileList)):
        imgfile_abs = os.path.join(INPUT_DIR,fileList[idx])
        #print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'index:', idx, 'Processing data, please wait...'
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imgfile_abs))
        output = net.forward()
        featureData = net.blobs[FEATURE_LAYER].data
        featureFileName = fileList[idx].replace('.jpg','_prob.txt')
        featureSavePath = os.path.join(featureSavePathBase,featureFileName)
        #print('The predicted score for {} is {}'.format(imgfile_abs, score))
        np.savetxt(featureSavePath, featureData)
        if idx % 100 == 0:
            print('Have processing [%d] images, Please waiting.....' % idx)
    print('Finished Extract Features')

def readLabelList(FILE_NAME_LIST):
    # read filename path and label, save them to list
    fileName = []
    with open(FILE_NAME_LIST, 'r') as fi:
        while(True):
            line = fi.readline().strip().split()
            if not line:
                break
            fileName.append(line[0])

    print('read fileList done, total num :', len(fileName))

    return fileName

def main():
    net = initilize()
    imgFilenameListTRAIN = readLabelList(TOTAL_FILELIST_CNN)
    #featureExtract(imgFilenameListTRAIN, net, FEATURE_SINGLE_DIR)
    featureExtract(imgFilenameListTRAIN, net, './data/cnn_prob/single')
    print('================================================')

if __name__ == '__main__':
    main()
