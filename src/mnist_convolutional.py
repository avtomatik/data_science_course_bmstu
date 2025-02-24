#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 00:11:57 2021
Created on Tue Nov  1 18:31:37 2022

@author: Nata; green-machine
"""

import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
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

num_filters = 8
kernel_size = 3
pool_size = 2

# =============================================================================
# Build the model.
# =============================================================================
model = Sequential(
    [
        Conv2D(num_filters, kernel_size, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size),
        Dropout(.5),
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
    epochs=5,
    validation_data=(test_images, to_categorical(test_labels))
)

model.save_weights(MODELS_DIR.joinpath('mnist_convolutional.h5'))

# =============================================================================
# Predict on the first 5 test images.
# =============================================================================
predictions = model.predict(test_images[:5])

# =============================================================================
# Print our model's predictions.
# =============================================================================
# =============================================================================
# [7 2 1 0 4]
# =============================================================================
print(np.argmax(predictions, axis=1))

# =============================================================================
# Check our predictions against the ground truths.
# =============================================================================
# =============================================================================
# [7 2 1 0 4]
# =============================================================================
print(test_labels[:5])

print(model.summary())
