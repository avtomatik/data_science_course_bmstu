#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:16:28 2022

@author: green-machine
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import np_utils

(_X_train, _X_test), (_y_train, _y_test) = load_data()
plt.imshow(Image.fromarray(_X_test[100]).convert('RGBA'))
plt.show()

X_train = _X_train.reshape(_X_train.shape[0], np.prod(_X_train.shape[1:]))
X_test = _X_test.reshape(_X_test.shape[0])

X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)

y_train = np_utils.to_categrical(_y_train, 10)
y_test = np_utils.to_categrical(_y_test, 10)

print(tensorflow.__version__)
