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

from constants import COMBINE_FEATURE_DIR_TRAIN
from constants import COMBINE_FEATURE_DIR_VAL
from constants import FEATURE_DIM
from constants import FACE_MAX
from constants import CLIP_FRAME_NUMBER
from constants import LSTM_HIDDEN_STATE
from constants import OVERLAP_STEP
from constants import RNN_MODEL_JSON_PATH
from constants import RNN_MODEL_JSON_PATH_GPU
from constants import CROSSVAL_FEATURE_DIR
from constants import COMBINE_FEATURE_DIR

def dataReadFromH5(trainTotal,valTotal):
    trainPath = os.path.join(COMBINE_FEATURE_DIR,trainTotal)
    valPath = os.path.join(COMBINE_FEATURE_DIR, valTotal)
    with h5py.File(trainPath, 'r') as f:
        X_train = f['data'][:]
        Y_train = f['grtru'][:]
        f.close
    with h5py.File(valPath, 'r') as f:
        X_test = f['data'][:]
        Y_test = f['grtru'][:]
        f.close

    sampleNumberTrain = X_train.shape[0]
    sampleNumberVal = X_test.shape[0]

    face1_train = X_train[:,0,:,:]
    face1_train.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face2_train = X_train[:,1,:,:]
    face2_train.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face3_train = X_train[:,2,:,:]
    face3_train.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face4_train = X_train[:,3,:,:]
    face4_train.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face5_train = X_train[:,4,:,:]
    face5_train.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face6_train = X_train[:,5,:,:]
    face6_train.shape = (sampleNumberTrain,CLIP_FRAME_NUMBER,FEATURE_DIM)

    face1_test = X_test[:,0,:,:]
    face1_test.shape = (sampleNumberVal,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face2_test = X_test[:,1,:,:]
    face2_test.shape = (sampleNumberVal,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face3_test = X_test[:,2,:,:]
    face3_test.shape = (sampleNumberVal,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face4_test = X_test[:,3,:,:]
    face4_test.shape = (sampleNumberVal,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face5_test = X_test[:,4,:,:]
    face5_test.shape = (sampleNumberVal,CLIP_FRAME_NUMBER,FEATURE_DIM)
    face6_test = X_test[:,5,:,:]
    face6_test.shape = (sampleNumberVal,CLIP_FRAME_NUMBER,FEATURE_DIM)

    X_train_rnn = [ face1_train,face2_train,face3_train,face4_train,face5_train,face6_train]
    X_test_rnn = [ face1_test,face2_test,face3_test,face4_test,face5_test,face6_test]


    Y_train.shape = (sampleNumberTrain,FACE_MAX,CLIP_FRAME_NUMBER)
    Y_train_rnn = np.transpose(Y_train,(0,2,1))

    Y_test.shape = (sampleNumberVal,FACE_MAX,CLIP_FRAME_NUMBER)
    Y_test_rnn = np.transpose(Y_test,(0,2,1))

    return X_train_rnn, Y_train_rnn, X_test_rnn, Y_test_rnn


def dataRead():
    totalFeatureTrain = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, FEATURE_DIM])
    totalGrTruTrain = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, 1])

    totalFeatureVal = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, FEATURE_DIM])
    totalGrTruVal = np.zeros([1,FACE_MAX, CLIP_FRAME_NUMBER, 1])

    h5FilenameListTrain = []
    h5FilenameListVal = []
    for (dirpath, dirnames, filenames) in os.walk(COMBINE_FEATURE_DIR_TRAIN):
        for filename in filenames:
            h5FilenameListTrain.append(filename)

    for (dirpath, dirnames, filenames) in os.walk(COMBINE_FEATURE_DIR_VAL):
        for filename in filenames:
            h5FilenameListVal.append(filename)

    for item in h5FilenameListTrain:
        print(item)
        path = os.path.join(COMBINE_FEATURE_DIR_TRAIN,item)
        with h5py.File(path, 'r') as f:
            clipFeatureMultiFace = np.array([f['data']])
            clipGrTruMultiFace = np.array([f['grtru']])
            f.close()
        totalFeatureTrain = np.vstack([totalFeatureTrain, clipFeatureMultiFace])
        totalGrTruTrain = np.vstack([totalGrTruTrain, clipGrTruMultiFace])

    for item in h5FilenameListVal:
        print(item)
        path = os.path.join(COMBINE_FEATURE_DIR_VAL,item)
        with h5py.File(path, 'r') as f:
            clipFeatureMultiFace = np.array([f['data']])
            clipGrTruMultiFace = np.array([f['grtru']])
            f.close()
        totalFeatureVal = np.vstack([totalFeatureVal, clipFeatureMultiFace])
        totalGrTruVal = np.vstack([totalGrTruVal, clipGrTruMultiFace])

    totalFeatureTrain = totalFeatureTrain[1:]
    totalGrTruTrain = totalGrTruTrain[1:]

    totalFeatureVal = totalFeatureVal[1:]
    totalGrTruVal = totalGrTruVal[1:]

    print(totalFeatureTrain.shape, totalGrTruTrain.shape)
    print(totalFeatureVal.shape, totalGrTruVal.shape)

    return totalFeatureTrain, totalGrTruTrain, totalFeatureVal, totalGrTruVal

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
    model.compile(optimizer='rmsprop',
              loss='mse')

    json_string = model.to_json()
    open(RNN_MODEL_JSON_PATH,'w').write(json_string)

    #plot(model, show_shapes=True,to_file='model2.png')
    print(model.summary())
    return model


def trainRNN(finetuning, DropoutW,DropoutU,model, epoches, batch_size,X_train, Y_train, X_test, Y_test, validation_split):
    if finetuning:
        print('Using fine tuning mode.')
        model.load_weights('./data/weight/rnn_weight_saved_overlap{}_newdataset_DW{}_DU{}_rmsprop.h5'.format(OVERLAP_STEP, DropoutW, DropoutU))

    modes = ['GPU']
    results = []
    #tensorBoard = TensorBoard(log_dir='./logs/{}'.format(CrossID+'_adam_sgd'), histogram_freq=10, write_graph=True)
    tensorBoard = TensorBoard(log_dir='./logs/experiment_1/', histogram_freq=10, write_graph=True)
    #checkpointer = ModelCheckpoint('./data/weight/rnn_weight_saved_overlap{}_adam_{}_sgd.h5'.format(OVERLAP_STEP,CrossID),verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint('./data/weight/rnn_weight_saved_overlap{}_newdataset_DW{}_DU{}_rmsprop.h5'.format(OVERLAP_STEP, DropoutW, DropoutU),verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    start_time = time.time()
    history = model.fit(X_train,Y_train , batch_size=batch_size, nb_epoch=epoches ,validation_split = validation_split,shuffle = True, callbacks=[tensorBoard,checkpointer,earlystopping])

    average_time_per_epoch = (time.time() - start_time) / epoches
    results.append((history, average_time_per_epoch))

    #===========plot
    plt.style.use('ggplot')
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_title('Accuracy')
    ax1.set_ylabel('Train Loss')
    ax1.set_xlabel('Epochs')
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Loss')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xlabel('Epochs')
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.set_title('Time')
    ax3.set_ylabel('Seconds')
    for mode, result in zip(modes, results):
        ax1.plot(result[0].epoch, result[0].history['loss'], label=mode)
        ax2.plot(result[0].epoch, result[0].history['val_loss'], label=mode)
    ax1.legend()
    ax2.legend()
    ax3.bar(np.arange(len(results)), [x[1] for x in results],
            tick_label=modes, align='center')
    plt.tight_layout()
    plt.savefig('./data/figure/loss_curve_rmsprop_DW{}_DU{}.png'.format(DropoutW, DropoutU), dpi=200)
    #plt.show()

def testRNN(DropoutW, DropoutU, batch_size, X_test, Y_test):
    model = model_from_json(open(RNN_MODEL_JSON_PATH).read())
    model.compile(optimizer='rmsprop',loss='mse')

    # model.load_weights('./data/weight/rnn_weight_saved_overlap{}_adam_{}.h5'.format(OVERLAP_STEP,CrossID))
    model.load_weights('./data/weight/rnn_weight_saved_overlap{}_newdataset_DW{}_DU{}_rmsprop.h5'.format(OVERLAP_STEP, DropoutW, DropoutU))
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('DropoutW:{}, DropoutU:{}, Test score:{}'.format(DropoutW, DropoutU,score))


if __name__ == '__main__':
    mode = sys.argv[1]
    if sys.argv[2] == '1':
        finetuning = True
    elif sys.argv[2] == '0':
        finetuning = False
    else:
        raise Exception('Fine tuning model error!')


    DropoutW = float(sys.argv[3])
    DropoutU = float(sys.argv[4])
    trainh5FilePath = 'trainTotalRNN_overlap{}.h5'.format(OVERLAP_STEP)
    valh5FilePath = 'testTotalRNN_overlap{}.h5'.format(OVERLAP_STEP)

    X_train, Y_train, X_test, Y_test = dataReadFromH5(trainh5FilePath, valh5FilePath)

    if mode == 'train':
        model = rnn_model(DropoutW,DropoutU,)
        trainRNN(finetuning, DropoutW,DropoutU,model,500,64,X_train, Y_train, X_test, Y_test, 0.20)
    elif mode == 'test':
        testRNN(DropoutW,DropoutU,64,X_test,Y_test)
    else:
        raise Exception('Train or test mode Error!!!!')
    #model.fit(X_train,Y_train , batch_size=batch_size, nb_epoch=epoches ,validation_split = 0.10,shuffle = True)
