from keras.preprocessing.image import list_pictures, load_img, img_to_array
import numpy as np
from scipy.io import loadmat
import os
import h5py
import random

img_rows, img_cols = 224, 224
data_path = "D:\Contest\practice\Oxford-flowers\datasets\image.h5"

def load_data():
    if os.path.exists(data_path):
        file = h5py.File(data_path, 'r')
        x_train = file["x_train"][:]
        file.close()
    else:
        files = list_pictures("D:\Contest\practice\Oxford-flowers\datasets\jpg")
        files.sort()
        x_train = []
        for f in files:
            x_train.append(img_to_array(load_img(f, False, (img_rows, img_cols))))
        x_train = np.array(x_train)
        image = h5py.File(data_path, "w")
        image.create_dataset("x_train", data=x_train)
        image.close()

    y_train = loadmat("D:\Contest\practice\Oxford-flowers\datasets\imagelabels.mat")["labels"][0]
    y_train = np.array(y_train)

    print("images downloaded!")
    return x_train, y_train

def split_dataset(x, y, factor):
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    ind = int(factor*len(x))
    x_train, x_validation = x[:ind], x[ind:]
    y_train, y_validation = y[:ind], y[ind:]
    return (x_train, y_train), (x_validation, y_validation)