#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:23:08 2022

@author: green-machine
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# from google.colab import drive
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Input, MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# from tensorflow.keras.utils import img_to_array
# %matplotlib inline


def dice_factor(y_true, y_pred):
    return (2 * K.sum(y_true * y_pred) + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)


# drive.mount('content/drive')

DIR_1 = '..data/raw/drive-download-20220929T084853Z-001/images_prepped_train'
DIR_2 = '..data/raw/drive-download-20220929T084853Z-001/images_prepped_test'
DIR_3 = '..data/raw/drive-download-20220929T084853Z-001/annotations_prepped_train'
DIR_4 = '..data/raw/drive-download-20220929T084853Z-001/annotations_prepped_test'

train_im = []
for filename in sorted(os.listdir(DIR_1)):
    train_im.append(
        image.load_img(Path(DIR_1).joinpath(filename), target_size=(88, 120))
    )

test_im = []
for filename in sorted(os.listdir(DIR_2)):
    test_im.append(
        image.load_img(Path(DIR_2).joinpath(filename), target_size=(88, 120))
    )

train_seg = []
for filename in sorted(os.listdir(DIR_3)):
    train_seg.append(
        image.load_img(Path(DIR_3).joinpath(filename), target_size=(88, 120))
    )

test_seg = []
for filename in sorted(os.listdir(DIR_4)):
    test_seg.append(
        image.load_img(Path(DIR_4).joinpath(filename), target_size=(88, 120))
    )

X_train = np.array([image.img_to_array(img) for img in train_im])
X_test = np.array([image.img_to_array(img) for img in test_im])

y_train = np.array([image.img_to_array(img) for img in train_seg])
y_test = np.array([image.img_to_array(img) for img in test_seg])


def index2color(index):
    color = (255, 255, 255)

    color_map = {
        0: (200, 0, 0),
        1: (0, 200, 0),
        2: (0, 0, 200),
        3: (200, 200, 0),
        4: (200, 0, 200),
        5: (0, 200, 200),
        6: (200, 200, 200),
        7: (100, 0, 0),
        8: (0, 100, 0),
        9: (0, 0, 100),
        10: (100, 100, 0),
        11: (100, 0, 100),
    }

    return color_map.get(index, color)


def color(dataset):
    out = []
    for pr in dataset:
        curr_pr = pr.copy()
        curr_matrix = []
        for j in range(curr_pr.shape[0]):
            curr_str = []
            for i in range(curr_pr.shape[1]):
                curr_str.append(index2color(curr_pr[j][i][0]))
            curr_matrix.append(curr_str)
        out.append(curr_matrix)
    return np.array(out).astype('uint8')


def one_hot12(dataset):
    dataset12 = []

    for t in range(dataset.shape[0]):
        yy = dataset[t].copy()
        yy_new = []

        for j in range(yy.shape[0]):
            curr_yy_str = []
            for i in range(yy.shape[1]):
                curr_yy_str.append(utils.to_categorical(yy[j][i][0], 12))
            yy_new.append(curr_yy_str)

        yy_new = np.array(yy_new)
        dataset12.append(yy_new)

    dataset12 = np.array(dataset12)

    return dataset12


def one_hot3(dataset):
    dataset3 = []

    for t in range(dataset.shape[0]):
        yy = dataset[t].copy()
        yy_new = []

        for j in range(yy.shape[0]):
            curr_yy_str = []
            for i in range(yy.shape[1]):
                if yy[j][i][0] == 1:
                    data = 0
                elif yy[j][i][0] == 8:
                    data = 1
                elif yy[j][i][0] == 9:
                    data = 1
                elif yy[j][i][0] == 10:
                    data = 1
                else:
                    data = 4

                data = min(2, yy[j][i][0])

                curr_yy_str.append(utils.to_categorical(data, 3))
            yy_new.append(curr_yy_str)

        yy_new = np.array(yy_new)
        dataset3.append(yy_new)

    dataset3 = np.array(dataset3)

    return dataset3


def one_hot4(dataset):
    dataset4 = []

    for t in range(dataset.shape[0]):
        yy = dataset[t].copy()
        yy_new = []

        for j in range(yy.shape[0]):
            curr_yy_str = []
            for i in range(yy.shape[1]):
                if yy[j][i][0] == 1:
                    data = 0
                elif yy[j][i][0] == 8:
                    data = 1
                elif yy[j][i][0] == 9:
                    data = 2
                elif yy[j][i][0] == 10:
                    data = 2
                else:
                    data = 3

                curr_yy_str.append(utils.to_categorical(data, 4))
            yy_new.append(curr_yy_str)

        yy_new = np.array(yy_new)
        dataset4.append(yy_new)

    dataset4 = np.array(dataset4)

    return dataset4


def color4(dataset):
    out4 = []
    for pr in dataset:
        curr_pr = pr.copy()
        curr_matrix = []
        for j in range(curr_pr.shape[0]):
            curr_str = []
            for i in range(curr_pr.shape[1]):
                curr_str.append(index2color(np.argmax(curr_pr[j][i][0])))
            curr_matrix.append(curr_str)
        out4.append(curr_matrix)
    return np.array(out4).astype('uint8')


y_train_out = color(y_train)
y_test_out = color(y_test)


n = 100
plt.imshow(train_seg[n].convert('RGBA'))
plt.show()

img = Image.fromarray(y_train_out[n])
plt.imshow(img.convert('RGBA'))
plt.show()


y_train12 = one_hot12(y_train)
y_test12 = one_hot12(y_test)


y_train4 = one_hot4(y_train)
y_test4 = one_hot4(y_test)


y4_out = color4(y_train4)
y4_test_out = color4(y_test4)


def unet(num_classes, input_shape):
    img_input = Input(input_shape)

    # =========================================================================
    # Block 1
    # =========================================================================
    x = Conv2D(128, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block1_out = Activation('relu')(x)

    x = MaxPooling2D()(block1_out)

    # =========================================================================
    # Block 2
    # =========================================================================
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block2_out = Activation('relu')(x)

    x = MaxPooling2D()(block2_out)

    # =========================================================================
    # Up 1
    # =========================================================================
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # =========================================================================
    # Up 2
    # =========================================================================
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(img_input, x)

    model.compile(optimizer=Adam(lr=1e-03),
                  loss='categorical_crossentropy', metrics=[dice_factor])

    model.summary()

    return model


model_s = unet(4, (88, 120, 3))
hostory = model_s.fit(X_train, y_train4, epochs=30,
                      batch_size=5, validation_data=(X_test, y_test4))

f, axes = plt.subplots(10, 2, figsize=(15, 30))
n = 33
for i in range(10):
    img = Image.fromarray(X_train[n + i].astype('uint8'))
    axes[i, 0].imshow(img.convert('RGBA'))

    img = Image.fromarray(y4_out[n + i])
    axes[i, 0].imshow(img.convert('RGBA'))

plt.show()

# =============================================================================
# More Plotting Here
# =============================================================================
