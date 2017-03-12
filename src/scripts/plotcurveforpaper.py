from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import cPickle
import h5py
import time
import sys
import matplotlib
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


with h5py.File('./data/predictedH5/video66_grtru_predicted.h5', 'r') as f:
    predictedToSave = f['predicted'][:]
    groundtruthToSave = f['groundtruth'][:]
    f.close()

figureDPI = 100
width = 18
height = 4*3 + 10

# font = {'family' : 'Times New Roman',
#         'weight' : 'bold',
#         'size'   : 22}
#print(matplotlib.rcParams.keys())
matplotlib.rcParams[u'font.serif'] = 'Times New Roman'
# matplotlib.rc('font', **font)

plt.figure(figsize=(width, height))


#plt.legend((h1[0], h2[0]), ('measured current', 'calculated current'), loc=2)

plt.subplot(3,1,1)
plt.xlim(0,610)
plt.ylim(-0.1,1.1)
plt.ylabel('$w_{1,t}$', fontsize=40)
plt.xlabel('Frame $t$', fontsize=30)

plt.plot(predictedToSave[0,:].tolist(), linewidth= 2.5,color = 'r',label='Predicted')
plt.plot(groundtruthToSave[0,:].tolist(),  linewidth= 2.5,color = 'b',label = 'Groundtruth')
plt.legend(loc = 2,prop={'family':'Times New Roman','size':21})
# plt.grid(True)
plt.title('Saliency weight curve', fontsize=48)

plt.subplot(3,1,2)
plt.xlim(0,610)
plt.ylim(-0.1,1.1)
plt.ylabel('$w_{2,t}$', fontsize=48)
plt.xlabel('Frame $t$', fontsize=30)
plt.plot(predictedToSave[1,:].tolist(),  linewidth= 2.5,color = 'r',label='Predicted')
plt.plot(groundtruthToSave[1,:].tolist(),  linewidth= 2.5,color = 'b',label = 'Groundtruth')
plt.legend(loc = 2,prop={'family':'Times New Roman','size':21})
#plt.grid(True)

plt.subplot(3,1,3)
plt.xlim(0,610)

plt.ylim(-0.1,1.1)
plt.ylabel('$w_{3,t}$', fontproperties='Times New Roman', fontsize=44)
plt.xlabel('Frame $t$',fontproperties='Times New Roman', fontsize=30)
plt.plot(predictedToSave[2,:].tolist(),  linewidth= 2.5,color = 'r',label='Predicted')
plt.plot(groundtruthToSave[2,:].tolist(), linewidth= 2.5,color =  'b',label = 'Groundtruth')
plt.legend(loc = 2,prop={'family':'Times New Roman','size':21})
# plt.grid(True, which='major') #


plt.savefig('./data/figure/video66.eps', dpi=figureDPI)
# print('./data/figure/predicted/predictedFigure_{}.png'.format(videoID))
plt.close()
