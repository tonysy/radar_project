from __future__ import print_function
import os
import numpy as np
from src.config import config

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
