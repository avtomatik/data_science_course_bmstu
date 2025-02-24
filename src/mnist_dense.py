#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:31:37 2022

@author: green-machine
"""

import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from config import MODELS_DIR

(train_images, train_labels), (test_images, test_labels) = load_data()

# =============================================================================
# Normalize the images.
# =============================================================================
train_images = train_images / 255 - .5
test_images = test_images / 255 - .5

# =============================================================================
# Reshape the images.
# =============================================================================
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = Sequential(
    [
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Flatten(),
        Dense(10, activation='softmax'),
    ]
)

# =============================================================================
# Compile the model.
# =============================================================================
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =============================================================================
# Train the model.
# =============================================================================
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels))
)

model.save_weights(MODELS_DIR.joinpath('mnist_dense.h5'))

print(model.summary())
