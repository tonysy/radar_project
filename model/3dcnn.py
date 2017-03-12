from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from sklearn.model_selection import train_test_split


def file_dict():
    """
    rename all files from prefix+id.txt into id.txt. such as : backward1.txt -->00001.txt
    :return filename_dict: contains {'gesture name: file_list'}
    """
    path = './data_0225'
    filename_dict = {'backward':[],'forward':[],'rotate':[],'static':[]}
    for root, dirs, files in os.walk(path):
        for item in files:
            for key in filename_dict.keys():
                if key in root:
                    if key in item:
                        new_name = item.replace(key,'').strip().split('.')[0].zfill(5) + '.txt'
                        filename = os.path.join(root, new_name)
                        os.rename(os.path.join(root,item), filename)
                        filename_dict[key].append(filename)
                    else:
                        # print "Filename has been renamed!"
                        filename = os.path.join(root, item)
                        filename_dict[key].append(filename)
    for key in filename_dict.keys():
        filename_dict[key].sort()

    return filename_dict
#
def data_to_image(filename_dict):
    """
    convert txt data into [frames X width X height]
    :param filename_dict: filename dictionary
    :return total_data_list, total_label_list: pay attention, the output is np.adday() foramt, with shape (nb_samples, framse, width, height) in total_data_list,  and (nb_samples, 1) in total_label_list.
    """
    total_data_list = []
    total_label_list = []
    for label_idx, key in enumerate(filename_dict.keys()):
        total = len(filename_dict[key])
        print(key, total)
        idx = 0
        flag = True
        while flag:
            if idx < total:
                sample = []
                for i in range(14):
                    output = txt_to_wh_matrix(filename_dict[key][idx])
                    sample.append(output)
                    idx += 1
                total_data_list.append([np.array(sample)])
                total_label_list.append(label_idx)

            else:
                flag = False

    dataset_data = np.array(total_data_list)
    dataset_label = np.array(total_label_list).reshape((len(total_label_list),1))

    return dataset_data, dataset_label

def txt_to_wh_matrix(filename):
    """
    convert a txt data into [width, height] format.
    :param filename: txt file path
    :return output: [width, height] format data.
    """
    f = open(filename, 'r')

    lines = f.readlines()
    for idx in range(len(lines)):
        lines[idx] = lines[idx].strip()
    output = np.array(lines).reshape(32,120)

    return output

#==========Config===========
nb_frames = 14

nb_filters = [25, 25]
nb_conv1 = [5, 5, 5]
nb_pool1 = [2, 2, 2]
nb_conv2 = [3, 3, 3]
nb_pool2 = [2, 2, 2]

nb_classes = 4
batch_size = 16
nb_epoch = 100
# =========Dataset==========
filename_dict = file_dict()
dataset_data, dataset_label = data_to_image(filename_dict)
print(dataset_data.shape)
X_train = dataset_data
Y_train = np_utils.to_categorical(dataset_label, nb_classes)
# ========Preprocessing===
X_train = X_train.astype('float32')
X_train -= np.mean(X_train)
X_train /= np.max(X_train)
# ========3D CNN Model===

model = Sequential()
model.add(Convolution3D(nb_filter=nb_filters[0],
                        kernel_dim1=nb_conv1[0],
                        kernel_dim2=nb_conv1[1],
                        kernel_dim3=nb_conv1[2],
                        input_shape=(1,14,32,120),
                        activation='relu',
                        dim_ordering="th"))
print("Load Conv1")
model.add(MaxPooling3D(pool_size=(nb_pool1[0],nb_pool1[1],nb_pool1[2]),dim_ordering="th"))
print("Load pool1")
model.add(Convolution3D(nb_filter=nb_filters[1],
                        kernel_dim1=nb_conv2[0],
                        kernel_dim2=nb_conv2[1],
                        kernel_dim3=nb_conv2[2],
                        activation='relu',
                        dim_ordering="th"))
print("Load Conv2")
model.add(MaxPooling3D(pool_size=(nb_pool2[0],nb_pool2[1],nb_pool2[2]), dim_ordering="th"))
print("Load Conv2")
model.add(Flatten())
model.add(Dense(1024, init='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, init='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])
#==========Dataset Split==========
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_train, Y_train, test_size=0.4, random_state=4)

#plot(model, to_file='./model.png')
assert True, 'plot model'
#=========Train model==========
start_time = time.time()
history = model.fit(X_train_new,
                    y_train_new,
                    validation_data=(X_val_new, y_val_new),
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    shuffle=True)
average_time_per_epoch = (time.time() - start_time) / nb_epoch

#=========Evaluate the model===
score = model.evaluate(X_val_new,
                      y_val_new,
                      batch_size=batch_size)
results =[(history, average_time_per_epoch)]
modes = ['GPU']
plt.style.use('ggplot')
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.set_title('Accuracy')
ax1.set_ylabel('Validation Accuracy')
ax1.set_xlabel('Epochs')
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.set_title('Loss')
ax2.set_ylabel('Validation Loss')
ax2.set_xlabel('Epochs')
ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax3.set_title('Time')
ax3.set_ylabel('Seconds')
for mode, result in zip(modes, results):
    ax1.plot(result[0].epoch, result[0].history['val_acc'], label=mode)
    ax2.plot(result[0].epoch, result[0].history['val_loss'], label=mode)
ax1.legend()
ax2.legend()
ax3.bar(np.arange(len(results)), [x[1] for x in results],
        tick_label=modes, align='center')
plt.tight_layout()
plt.show()
