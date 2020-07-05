import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
import numpy as np

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir_list = sorted(os.listdir(path))
    print(image_dir_list)
    #x = np.zeros((len(image_dir), 256, 256, 3), dtype=np.uint8)
    #y = np.zeros((len(image_dir)), dtype=np.uint8)
    label_count = [0] * 42
    total_train_count = 0
    total_val_count = 0
    for i, dir in enumerate(image_dir_list):
        print("Preprocess dir {}".format(dir))
        img_dir = os.path.join(path, dir)
        img_list = sorted(os.listdir(img_dir))
        train_size = int(0.9*len(img_list))
        val_size = int(0.1*len(img_list))

        if i == 0:
            train_x = np.zeros((train_size, 256, 256, 3), dtype=np.uint8)
            train_y = np.zeros(train_size, dtype=np.uint8)
            val_x = np.zeros((val_size, 256, 256, 3), dtype=np.uint8)
            val_y = np.zeros(val_size, dtype=np.uint8)

        else:
            train_x = np.concatenate((train_x, np.zeros((train_size, 256, 256, 3), dtype=np.uint8)), axis=0)
            train_y = np.concatenate((train_y, np.zeros(train_size, dtype=np.uint8)))
            val_x = np.concatenate((val_x, np.zeros((val_size, 256, 256, 3), dtype=np.uint8)), axis=0)
            val_y = np.concatenate((val_y, np.zeros(val_size, dtype=np.uint8)))

        for j, file in enumerate(img_list):
            img = cv2.imread(os.path.join(img_dir, file))
            if j >= train_size + val_size:
                break

            if j < train_size:
                train_x[total_train_count+j, :, :] = cv2.resize(img,(256, 256))
                train_y[total_train_count+j] = int(image_dir_list[i])
            else:
                val_x[total_val_count+j-train_size, :, :] = cv2.resize(img,(256, 256)) 
                val_y[total_val_count+j-train_size] = int(image_dir_list[i])
            #if label:
                
        total_train_count += train_size
        total_val_count += val_size

    if label:
      return train_x, train_y, val_x, val_y
    else:
      return x

def read_test_file(path):
    image_list = sorted(os.listdir(path))
    print(len(image_list))
    x = np.zeros((len(image_list), 256, 256, 3), dtype=np.uint8)
    filename_list = []
    for i, file_name in enumerate(image_list):
        img = cv2.imread(os.path.join(path, file_name))
        x[i, :, :] = cv2.resize(img,(256, 256))
        filename_list.append(file_name)
    return x, filename_list


def set_seed(SEED):
    SEED = 0
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
