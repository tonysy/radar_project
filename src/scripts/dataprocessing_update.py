#!/usr/bin/env python
#coding:utf-8
"""
@author: Songyang Zhang
@licence: MIT Licence
@contact: syzhang@buaa.edu.cn

@version: 0.0.1

@note:
@attention:
@bug:
@warning:

"""

from __future__ import print_function
import numpy as np
import cPickle
import h5py
import os
import time
import pandas
import re

from constants import FEATURE_SINGLE_DIR_TRAIN
from constants import FEATURE_SINGLE_DIR_VAL
from constants import FACE_MAX
from constants import FEATURE_DIM
from constants import COMBINE_FEATURE_DIR_TRAIN
from constants import COMBINE_FEATURE_DIR_VAL
from constants import REGRESSION_DIR_TRAIN
from constants import REGRESSION_DIR_VAL
from constants import CLIP_FRAME_NUMBER
from constants import TRAIN_FILELIST_CNN
from constants import VAL_FILELIST_CNN
from constants import OVERLAP_STEP
from constants import TOTAL_FILELIST_CNN
from constants import FEATURE_SINGLE_DIR
from constants import COMBINE_FEATURE_DIR

def FeatureCombineOverlap(ClipFrameNumber,OverlapStep, combineDataBath, featureDataDir, regressionDic, fileListDic):
    """
    Combine all single feature file into dataset shape used for RNN-LSTM,
    For every video sample:
    [FrameNumber, FaceMax]
    Because of different frame length, we cut a video into serveral clip
    overlap on whole video.

    @param OverlapStep: use window to convert the whole video into serveral clips
    @param FrameClip: frame length of a clip
    @param FaceMax: max face number corresponce with RNN network
    """

    #print('All feature files name list, dataset shape is:' , fileList.shape)


    zeroFeature = np.zeros([1, ClipFrameNumber, FEATURE_DIM]) # use zeros feature to stand for no face
    zeroGrTru = np.zeros([1,ClipFrameNumber,1])
    videoIDList = fileListDic.keys()
    for ID in videoIDList:
        SampleFrameNumber = fileListDic[ID][0]
        SplitClipMax = (SampleFrameNumber - ClipFrameNumber) / OverlapStep # For every video sample, max clips can be convert to

        faceNumber = fileListDic[ID][1]
        zeroFaceNumber = FACE_MAX - faceNumber # zero vetor number
        ID = int(ID)
        for clipIdx in range(SplitClipMax):
            clipFeatureMultiFace = np.zeros([1,ClipFrameNumber, FEATURE_DIM])
            clipGrTruMultiFace = np.zeros([1,ClipFrameNumber,1])
            if zeroFaceNumber >= 0:
                for face in range(1,faceNumber+1):
                    # initial data shape
                    singleFaceClipFeature = np.empty([1,FEATURE_DIM])  #featureDim: 128 or 1024
                    #print('single', singleFaceClipFeature.shape)
                    singleFaceClipGrTru = np.empty([1,1]) # regression groundtruth

                    clipStart = 10 + (clipIdx)*OverlapStep
                    clipEnd = clipStart + ClipFrameNumber

                    for frameIdx in range(clipStart, clipEnd):
                        featurePathAbs = featureFileAbsGet(ID, frameIdx, face,featureDataDir)
                        featureTemp = np.loadtxt(featurePathAbs) # get feature
                        featureTemp.shape = (128,1)
                        featureTemp = np.transpose(featureTemp)
                        #print(featureTemp.shape)

                        grtruKey = 'video{}_frame{}_face{}.jpg'.format(ID, frameIdx, face)
                        grtruKey = regressionDic[grtruKey]
                        grtruKeyTemp = np.array([grtruKey])
                        grtruKeyTemp.shape = (1,1)# get regression groundtruth

                        singleFaceClipFeature = np.vstack([singleFaceClipFeature,featureTemp])
                        singleFaceClipGrTru = np.vstack([singleFaceClipGrTru, grtruKeyTemp])


                    singleFaceClipFeature = singleFaceClipFeature[1:]
                    singleFaceClipGrTru = singleFaceClipGrTru[1:]
                    singleFaceClipFeature = np.array([singleFaceClipFeature]) # convert clip feature shape into (1,clipFrame, feture_dim)
                    #print('aaa',singleFaceClipFeature.shape)
                    singleFaceClipGrTru = np.array([singleFaceClipGrTru])


                    clipFeatureMultiFace = np.vstack([clipFeatureMultiFace, singleFaceClipFeature])
                    clipGrTruMultiFace = np.vstack([clipGrTruMultiFace, singleFaceClipGrTru])

                for i in range(zeroFaceNumber):
                    clipFeatureMultiFace = np.vstack([clipFeatureMultiFace, zeroFeature])
                    clipGrTruMultiFace = np.vstack([clipGrTruMultiFace, zeroGrTru])

                clipFeatureMultiFace = clipFeatureMultiFace[1:]
                clipGrTruMultiFace  = clipGrTruMultiFace[1:]
                print('Video ID:', ID, ', Clip Index:',clipIdx,' ,Feature shape is:',clipFeatureMultiFace.shape, 'GroundTruth shape is:', clipGrTruMultiFace.shape)

                if not os.path.exists(os.path.join(combineDataBath,'video{}'.format(ID))):
                    os.makedirs(os.path.join(combineDataBath,'video{}'.format(ID)))
                    print('Create video{} Successfully!'.format)
                clipMultiFaceSaveFilename = 'video{}/combine_video{}_clip{}_featuredim{}.h5'.format(ID,ID,clipIdx,FEATURE_DIM)
                clipMultiFaceSavePath = os.path.join(combineDataBath,clipMultiFaceSaveFilename)

                with h5py.File(clipMultiFaceSavePath, 'w') as f:
                    f['data'] = clipFeatureMultiFace
                    f['grtru'] = clipGrTruMultiFace
                    f.close()
                    print('Save Successfully!')

def readFilenameList(filenameList_FilePath):
    # read filename path and label, save them to list
    fileName = []
    fileGrtru = []
    with open(filenameList_FilePath, 'r') as fi:
        while(True):
            line = fi.readline().strip().split()
            if not line:
                break
            fileName.append(line[0])
            fileGrtru.append(float(line[1]))
    print('read fileList done, total num :', len(fileName))

    grtruDicTemp = zip(fileName, fileGrtru)
    grtruDic = dict((key,value) for key, value in grtruDicTemp)

    return fileName , grtruDic

def filenameConvert(fileName):
    # convert filename into index(include: video ID, frame ID, face ID)
    fileList = []
    for i in range(len(fileName)):
        itemTemp = []
        videoNumber = re.findall('(\d+)_frame',fileName[i])
        videoNumber = int(videoNumber[0])
        frameNumber = re.findall('_frame(\d+)',fileName[i])
        frameNumber = int(frameNumber[0])
        faceNumber = re.findall('_face(\d+)',fileName[i])
        faceNumber = int(faceNumber[0])
        #slabel = fileLabel[i]
        itemTemp.append(videoNumber)
        itemTemp.append(frameNumber)
        itemTemp.append(faceNumber)
        #itemTemp.append(label)
        #print(itemTemp)
        fileList.append(itemTemp)

    fileList = np.array(fileList)

    print(fileList.shape)
    videoIDList = np.unique(fileList[:,0]).tolist()
    fileListDic = {}
    for item in videoIDList:
        fileListDic[item] = [0,0]
    #print(fileListDic)

    for idx in range(len(fileName)):
        frameNb = fileListDic[fileList[idx,0]][0]
        faceNb = fileListDic[fileList[idx,0]][1]

        currentFrameNb = fileList[idx][1]
        currentFaceNb = fileList[idx][2]
        if frameNb < currentFrameNb:
            fileListDic[fileList[idx,0]][0] = currentFrameNb
        if faceNb < currentFaceNb:
            fileListDic[fileList[idx,0]][1] = currentFaceNb
    print(fileListDic)

    return fileList, fileListDic

def featureFileAbsGet(videoID, frameNb, faceID, FEATURE_DATA_DIR):

    featureFileName = 'video'+str(videoID)+'_frame'+str(frameNb)+'_face'+str(faceID)+'_fc.txt'
    featurePathAbs = os.path.join(FEATURE_DATA_DIR, featureFileName)
    return featurePathAbs



def main():
#    fileName = readFilenameList(TRAIN_FILELIST_CNN)
#    fileList, fileListDic = filenameConvert(fileName)
#    FeatureCombineOverlap(ClipFrameNumber=CLIP_FRAME_NUMBER,OverlapStep=OVERLAP_STEP, combineDataBath=COMBINE_FEATURE_DIR_TRAIN, featureDataDir=FEATURE_SINGLE_DIR_TRAIN, regressionDir=REGRESSION_DIR_TRAIN, fileListDic = fileListDic)

    fileName, grtruDic = readFilenameList(TOTAL_FILELIST_CNN)
    fileList, fileListDic = filenameConvert(fileName)
    FeatureCombineOverlap(ClipFrameNumber=CLIP_FRAME_NUMBER,OverlapStep=OVERLAP_STEP, combineDataBath=COMBINE_FEATURE_DIR, featureDataDir=FEATURE_SINGLE_DIR, regressionDic= grtruDic, fileListDic = fileListDic)

if __name__ == '__main__':
    main()
