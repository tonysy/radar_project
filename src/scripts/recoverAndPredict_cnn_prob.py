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
from keras.models import model_from_json


from constants import COMBINE_FEATURE_DIR_TRAIN
from constants import COMBINE_FEATURE_DIR_VAL
from constants import FEATURE_DIM
from constants import FACE_MAX
from constants import CLIP_FRAME_NUMBER
from constants import LSTM_HIDDEN_STATE
from constants import RNN_WEIGHT_PATH
from constants import OVERLAP_STEP
from constants import PREDICTED_SAVE_DIR
from constants import TRAIN_FILELIST_CNN
from constants import VAL_FILELIST_CNN
from constants import RNN_MODEL_JSON_PATH_GPU
from constants import FILE_NAME_DICT
from constants import CROSSVAL_FEATURE_DIR
from constants import CROSS_GROUP_DICT
from constants import TOTAL_FILELIST_CNN
from constants import RNN_WEIGHT_PATH_CROSSVAL
#from constants import FEATURE_COMBINE_RNN_DIR
from dataprocessing import readFilenameList
from dataprocessing import filenameConvert

def dataReadForPredicted(videoID, clipNumberTotal,combine_dir ):
    totalFeature = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, FEATURE_DIM])
    totalGrTru = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, 1])
    totalCNN = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, 1])

    for i in range(clipNumberTotal):
        combineClipFileName = 'combine_{}_clip{}_featuredim128.h5'.format(videoID,i)
        filePath = os.path.join(combine_dir,videoID,combineClipFileName)
        #print(filePath)
        with h5py.File(filePath, 'r') as f:
            clipFeatureMultiFace = np.array([f['data']])
            clipGrTruMultiFace = np.array([f['grtru']])
            f.close()

        CNNProbfilePath = os.path.join('./data/cnn_prob/combine_1',videoID,combineClipFileName)
        #print(filePath)
        with h5py.File(CNNProbfilePath, 'r') as f:
            clipCNNMultiFace = np.array([f['grtru']])
            f.close()

        totalFeature = np.vstack([totalFeature, clipFeatureMultiFace])
        totalGrTru = np.vstack([totalGrTru, clipGrTruMultiFace])
        totalCNN = np.vstack([totalCNN,clipCNNMultiFace])

    totalFeature = totalFeature[1:]
    totalGrTru = totalGrTru[1:]
    totalCNN = totalCNN[1:]

    print(totalFeature.shape, totalGrTru.shape,totalCNN.shape)

    X = totalFeature
    Y = totalGrTru
    Z = totalCNN
    sampleNumberTrain = X.shape[0]

    face1 = X[:,0,:,:]
    face1.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face2 = X[:,1,:,:]
    face2.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face3 = X[:,2,:,:]
    face3.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face4 = X[:,3,:,:]
    face4.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face5 = X[:,4,:,:]
    face5.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face6 = X[:,5,:,:]
    face6.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)

    X_rnn = [ face1,face2,face3,face4,face5,face6]

    Y.shape = (sampleNumberTrain,FACE_MAX,CLIP_FRAME_NUMBER)
    Y_rnn = np.transpose(Y,(0,2,1))

    Z.shape = (sampleNumberTrain,FACE_MAX,CLIP_FRAME_NUMBER)
    Z_rnn = np.transpose(Z,(0,2,1))

    return X_rnn, Y_rnn,Z_rnn

def recoverToVideo(recover_mode,overlap_step, videoID,faceNumber,clipNumberTotal, videoPredictedResults):

    startNineFrame_Face = np.ones([9, faceNumber]) / faceNumber
    startNineFrame_NoFace = np.zeros([9, FACE_MAX - faceNumber])
    startNineFrame = np.hstack([startNineFrame_Face, startNineFrame_NoFace])
    #print(startNineFrame)

    # every sample can be divide into serveral overlap
    ClipDivideOverlap = CLIP_FRAME_NUMBER / overlap_step
    #print('ClipDivideOverlap:',ClipDivideOverlap)
    # video recoved frame number from RNN network
    recoverVideoFrameNumber = CLIP_FRAME_NUMBER +  clipNumberTotal* overlap_step
    #print('recoverVideoFrameNumber:',recoverVideoFrameNumber)
    # divide recoverVideoFrame into sequence by overlap step length
    sequenceNumber = recoverVideoFrameNumber / overlap_step
    #print('sequenceNumber:', sequenceNumber)
    if recover_mode:
        #recover_mode: True====> use average mode to recover the whole video

        # clipTotal: predicted with model by X_rnn as input
        clipTotal = np.zeros([1,FACE_MAX])

        print('videoPredictedResults shape:', videoPredictedResults.shape)
        for seqIdx in range(sequenceNumber):
            # every overlap_step length called sequence
            #print('seqIdx:',seqIdx)
            #sequenceDataAverage = videoPredictedResults[seqIdx,:overlap_step,:]
            sumTimes = 0
            sequenceDataAverageStart = np.zeros([1,overlap_step,FACE_MAX])
            for idx in range(seqIdx+1):

                startIdx = (seqIdx - idx) * overlap_step
                endIdx = startIdx + overlap_step
                if idx >= clipNumberTotal:
                    break
                if (startIdx < CLIP_FRAME_NUMBER):
                    sumTimes += 1
                    tempdata = videoPredictedResults[idx,startIdx:endIdx, :]
                    sequenceDataAverageStart = np.add(sequenceDataAverageStart, tempdata)

            sequenceDataAverage = sequenceDataAverageStart / sumTimes
            sequenceDataAverage.shape = (overlap_step, FACE_MAX)
            #print(clipTotal.shape)

            clipTotal = np.vstack([clipTotal, sequenceDataAverage])

        clipTotal = clipTotal[1:]
        #print('clipTotal shape:', clipTotal.shape)
        clipTotal = np.vstack([startNineFrame,clipTotal])
        #print('clipTotal add 9 frame',clipTotal.shape)
        clipTotal = np.transpose(clipTotal)
        #print('clipTotal transpose shape:', clipTotal.shape)

    return clipTotal

def recoverToVideoNoAverage(faceNumber,groundtruth, overlap_step):

    startNineFrame_Face = np.ones([9, faceNumber]) / faceNumber
    startNineFrame_NoFace = np.zeros([9, FACE_MAX - faceNumber])
    startNineFrame = np.hstack([startNineFrame_Face, startNineFrame_NoFace])

    clip_number = groundtruth.shape[0]
    clipTotal = np.zeros([1,FACE_MAX])
    for i in range(clip_number-1):
        tempdata = groundtruth[i,:overlap_step,:]
        tempdata.shape= (overlap_step,FACE_MAX)
        clipTotal = np.vstack([clipTotal, tempdata])

    tempdatalastone = groundtruth[clip_number-1,:,:]
    tempdatalastone.shape= (CLIP_FRAME_NUMBER,FACE_MAX)
    clipTotal = np.vstack([clipTotal, tempdatalastone])

    clipTotal = clipTotal[1:]
    #print('clipTotal shape:', clipTotal.shape)
    clipTotal = np.vstack([startNineFrame,clipTotal])
    #print('clipTotal add 9 frame',clipTotal.shape)
    clipTotal = np.transpose(clipTotal)
    #print('clipTotal transpose shape:', clipTotal.shape)

    return clipTotal

def rnn_model(DropoutW, DropoutU):

    input_face1 = Input(shape=(CLIP_FRAME_NUMBER,FEATURE_DIM), name = 'face_input_1')
    input_face2 = Input(shape=(CLIP_FRAME_NUMBER,FEATURE_DIM), name = 'face_input_2')
    input_face3 = Input(shape=(CLIP_FRAME_NUMBER,FEATURE_DIM), name = 'face_input_3')
    input_face4 = Input(shape=(CLIP_FRAME_NUMBER,FEATURE_DIM), name = 'face_input_4')
    input_face5 = Input(shape=(CLIP_FRAME_NUMBER,FEATURE_DIM), name = 'face_input_5')
    input_face6 = Input(shape=(CLIP_FRAME_NUMBER,FEATURE_DIM), name = 'face_input_6')

    shared_lstm = LSTM(LSTM_HIDDEN_STATE, return_sequences = True, dropout_W = DropoutW, dropout_U=DropoutU)

    output_lstm_face1 = shared_lstm(input_face1)
    output_lstm_face2 = shared_lstm(input_face2)
    output_lstm_face3 = shared_lstm(input_face3)
    output_lstm_face4 = shared_lstm(input_face4)
    output_lstm_face5 = shared_lstm(input_face5)
    output_lstm_face6 = shared_lstm(input_face6)

    predict_out_1 = TimeDistributed(Dense(1, activation= 'relu'), name = 'predict_out_1')(output_lstm_face1)
    predict_out_2 = TimeDistributed(Dense(1, activation= 'relu'), name = 'predict_out_2')(output_lstm_face2)
    predict_out_3 = TimeDistributed(Dense(1, activation= 'relu'), name = 'predict_out_3')(output_lstm_face3)
    predict_out_4 = TimeDistributed(Dense(1, activation= 'relu'), name = 'predict_out_4')(output_lstm_face4)
    predict_out_5 = TimeDistributed(Dense(1, activation= 'relu'), name = 'predict_out_5')(output_lstm_face5)
    predict_out_6 = TimeDistributed(Dense(1, activation= 'relu'), name = 'predict_out_6')(output_lstm_face6)

    x = merge([predict_out_1, predict_out_2, predict_out_3, predict_out_4, predict_out_5, predict_out_6], mode='concat', name = 'merge_total')
    total_loss = TimeDistributed(Dense(FACE_MAX, activation='softmax'), name = 'total_output')(x)

    model = Model(input = [input_face1,input_face2,input_face3,input_face4,input_face5, input_face6],
                  output = total_loss)
    model.compile(optimizer='adam',
              loss='mse')


    print(model.summary())
    return model

def predictWithRNN(model,X_RNN, Y_RNN,Z):

    predictedResults = model.predict(X_RNN)
    groundtruth = Y_RNN
    cnnprob = Z
    return predictedResults, groundtruth,cnnprob


def SaveAndPlotFigure(videoID, faceNumber,recoverVideoPredicted, recoverVideoGrountruth,recoverVideoCNN ):

    figureDPI = 100
    width = 24
    height = 4*faceNumber + 1

    plt.figure(figsize=(width, height))


    #with h5py.File('./data/predictedH5/{}_faceNumberTotal{}_{}_file.h5'.format(videoID,faceNumber,CrossID), 'w') as f:
    with h5py.File('./data/cnn_prob/predicted_1/{}_cnn_prob.h5'.format(videoID), 'w') as f:
        # predictedToSave = recoverVideoPredicted[0:faceNumber]
        # print('Predicted shape:',predictedToSave.shape)
        # f['predicted'] = predictedToSave

        groundtruthToSave = recoverVideoCNN[0:faceNumber]
        print('Ground Truth shape:', groundtruthToSave.shape)
        f['cnnprob'] = groundtruthToSave
        f.close()

    for idx in range(faceNumber):
        plt.subplot(faceNumber,1,idx+1)
        plt.ylim(-0.1,1.1)
        plt.ylabel('Fixation Percentage')
        plt.xlabel('Frame Number')
        plt.title('{}: Face {} fixation curve'.format(videoID, idx+1))
        plt.plot(recoverVideoPredicted[idx,:].tolist(), 'r')
        plt.plot(recoverVideoGrountruth[idx,:].tolist(), 'b')
        plt.plot(recoverVideoCNN[idx,:].tolist(), 'k')

    plt.savefig('./data/cnn_prob/figure_1/predictedFigure_{}.png'.format(videoID), dpi=figureDPI)
    print('./data/cnn_prob/figure_1/predictedFigure_{}.png'.format(videoID))
    plt.close()



def predictAndPlot(DropoutW, DropoutU, videoIDList,fileListDic, overlap_step, combine_dir):
    model = rnn_model(DropoutW, DropoutU)
    #model = model_from_json(RNN_MODEL_JSON_PATH)
    model.load_weights('./data/weight/rnn_weight_saved_overlap{}_newdataset_DW{}_DU{}_adam.h5'.format(OVERLAP_STEP, DropoutW, DropoutU))

    for videoID in videoIDList: #videoID  string 'video1' 'video2'............
        IDnumber = int(videoID[5:])
        print(IDnumber)
        SampleFrameNumber = fileListDic[IDnumber][0]
        faceNumber = fileListDic[IDnumber][1]

        clipNumberTotal = (SampleFrameNumber - CLIP_FRAME_NUMBER) / overlap_step

        X_rnn, Y_rnn,Z_rnn = dataReadForPredicted(videoID, clipNumberTotal, combine_dir)

        videoPredictedResults, groundtruth,cnnprob = predictWithRNN(model, X_rnn, Y_rnn,Z_rnn)

        recoverVideoPredicted = recoverToVideo(True,OVERLAP_STEP, IDnumber, faceNumber, clipNumberTotal, videoPredictedResults)
        recoverVideoCNN = recoverToVideoNoAverage(faceNumber,cnnprob, OVERLAP_STEP)
        recoverVideoGrountruth = recoverToVideoNoAverage(faceNumber,groundtruth, OVERLAP_STEP)
        SaveAndPlotFigure(videoID, faceNumber, recoverVideoPredicted, recoverVideoGrountruth,recoverVideoCNN )
        #recoverVideoPredictedNoAverage = recoverToVideoNoAverage(faceNumber,videoPredictedResults,OVERLAP_STEP)
def main():
    fileName = readFilenameList(TOTAL_FILELIST_CNN)
    fileList, fileListDic = filenameConvert(fileName)

    DropoutW = float(sys.argv[1])
    DropoutU = float(sys.argv[2])
    videoIDList = []
    for (dirpath, dirnames,filenames) in os.walk(COMBINE_FEATURE_DIR_VAL):
        videoIDList = dirnames
        break

    predictAndPlot(DropoutW, DropoutU, videoIDList, fileListDic,OVERLAP_STEP,COMBINE_FEATURE_DIR_VAL)

    # data = np.loadtxt('./data/test_loss_total_overlap10.txt')
    # plt.plot(data)
    # plt.show()

if __name__ == '__main__':

    main()

    # for item in videoIDList:
    #     print(item)
    #     for (rootpath, subfolder, clipsH5) in os.walk(os.path.join(CROSSVAL_FEATURE_DIR,testDir,item)):
    #         print(clipsH5)
    #         break



    #main()
#    print(type(FILE_NAME_DICT))
#    print(FILE_NAME_DICT.keys())
