
FACE_MAX = 6 # MAX FACE NUMBER
FACE_MAX_2 = 9 # MAX FACE NUMBER

GPU_ID = 0
#caffe dir
CAFFE_ROOT_DIR = '/home/tony/caffe-master'
#mean.npy path
MEAN_FILE_PATH = './data/videofaceLYF_mean.npy'
#train filelist path
TRAIN_FILELIST_CNN = './data/train_classfication2.txt'
TOTAL_FILELIST_CNN = './data/all2_regression.txt'
#val filelist path
VAL_FILELIST_CNN = './data/val_classfication2.txt'
# deploy.prototxt
DEPLOY_TXT_PATH = './net/cnn/128_deploy_feature.prototxt'
# cnn model file path
#MODEL_FILE_PATH = './net/cnn/128_class2__iter_56788.caffemodel'
MODEL_FILE_PATH = './net/cnn/128_class2_165all__iter_95000.caffemodel'
# original picuture bath different
INPUT_DIR = './data/input'
# which layer feature need to extract
# FEATURE_LAYER = 'cls3_fc_lyf'
FEATURE_LAYER = 'prob'
# Feature single file bast dir
# FEATURE_SINGLE_DIR_TRAIN = './data/feature_extract/train'
# FEATURE_SINGLE_DIR_VAL = './data/feature_extract/val'
FEATURE_SINGLE_DIR = './data/featureSingleFigure'
FEATURE_SINGLE_DIR_TRAIN = './../cvpr2017_1024/data/feature_extract/train'
FEATURE_SINGLE_DIR_VAL = './../cvpr2017_1024/data/feature_extract/val'

#
# COMBINE_FEATURE_DIR_TRAIN = './data/feature_combine_forRNN/overlap10_clip100/train'
# COMBINE_FEATURE_DIR_VAL = './data/feature_combine_forRNN/overlap10_clip100/val'

COMBINE_FEATURE_DIR_TRAIN = './data/featureCombineForRNN/train'
COMBINE_FEATURE_DIR_VAL = './data/featureCombineForRNN/test'
COMBINE_FEATURE_DIR = './data/featureCombineForRNN'

REGRESSION_DIR_TRAIN = './../cvpr2017_1024/data/regressionGrTru/train'
REGRESSION_DIR_VAL = './../cvpr2017_1024/data/regressionGrTru/val'
REGRESSION_DIR = './data/singleFigureGrtru'
OVERLAP_STEP = 10

# RNN_WEIGHT_PATH = './data/rnnweight/rnn_weight_saved_300.h5'
#RNN_WEIGHT_PATH = './data/rnnweight/rnn_weight_saved_450_overlap{}.h5'.format(OVERLAP_STEP)
RNN_WEIGHT_PATH = './data/weight/rnn_weight_saved_overlap10_adam_cross1_sgd.h5'
RNN_WEIGHT_PATH_CROSSVAL = {'cross1':'./data/tempWeight/rnn_weight_saved_overlap10_adam_cross1_sgd.h5',
                            'cross2':'./data/tempWeight/rnn_weight_saved_overlap10_rmsprop_adam_cross2.h5',
                            'cross3':'./data/tempWeight/rnn_weight_saved_overlap10_rmsprop_adam_cross3.h5',
                            'cross4':'./data/tempWeight/rnn_weight_saved_overlap10_adam_cross4_sgd.h5',
                            'cross5':'./data/tempWeight/rnn_weight_saved_overlap10_adam_cross5_sgd.h5'}
CLIP_FRAME_NUMBER = 100
# OVERLAP_STEP = 20

FEATURE_DIM = 128
#LSTM_HIDDEN_STATE = 16
LSTM_HIDDEN_STATE = 32

PREDICTED_SAVE_DIR = './data/predictedSave/'
RNN_MODEL_JSON_PATH = './net/rnn/my_model_architecture_newdataset.json'
RNN_MODEL_JSON_PATH_GPU = './net/rnn/my_model_architecture.json'

CROSSVAL_FEATURE_DIR = './data/crossFeature'


CROSS_GROUP_DICT = {'cross1':['group1','group2','group3','group4', 'group5'],
                    'cross2':['group2','group1','group3','group4', 'group5'],
                    'cross3':['group3','group2','group1','group4', 'group5'],
                    'cross4':['group4','group2','group3','group1', 'group5'],
                    'cross5':['group5','group2','group3','group4', 'group1'],}
FILE_NAME_DICT = {1: [600, 2],
2: [493, 2],
3: [1440, 3],
4: [600, 2],
5: [500, 6],
6: [498, 3],
7: [500, 2],
8: [600, 3],
9: [480, 2],
10: [579, 4],
11: [523, 3],
12: [396, 4],
14: [485, 3],
15: [478, 5],
16: [500, 2],
18: [523, 2],
20: [600, 1],
21: [592, 2],
22: [476, 3],
23: [505, 3],
24: [516, 6],
25: [455, 3],
26: [521, 1],
27: [500, 3],
28: [600, 2],
30: [580, 6],
31: [500, 3],
33: [1440, 2],
34: [500, 3],
35: [467, 2],
36: [630, 2],
37: [627, 3],
40: [547, 6],
43: [630, 6],
44: [500, 2],
45: [630, 6],
47: [615, 5],
48: [500, 2],
49: [601, 4],
50: [630, 3],
53: [618, 3],
54: [1440, 2],
56: [492, 4],
57: [480, 2],
58: [615, 4],
60: [598, 5],
61: [500, 2],
62: [575, 4],
63: [500, 3],
64: [1440, 4],
65: [526, 2],
66: [616, 3],
67: [500, 2],
68: [500, 2],
69: [589, 2],
70: [500, 2],
71: [400, 2],
72: [600, 5],
73: [500, 2],
74: [600, 2],
75: [500, 2]
}
