#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:31:37 2022

@author: green-machine
"""

import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images / 255 - .5
test_images = test_images / 255 - .5

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

filters = 8
kernel_size = 3
pool_size = 2

model = Sequential(
    [
        Conv2D(filters, kernel_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size),
        Flatten(),
        Dense(10, activation='softmax'),
    ]
)

model.compile('adam', loss='categorical_crossentropy', metrics='accuracy')

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    validation_data=[test_images, to_categorical(test_labels)]
)

predictions = model.predict(test_images[:5])

# =============================================================================
# 7, 2, 1, 0, 4
# =============================================================================
print(np.argmax(predictions, axis=1))
print(test_labels[:5])
print(model.summary())
