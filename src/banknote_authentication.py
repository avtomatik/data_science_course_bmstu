#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:48:34 2022

@author: green-machine
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
df = pd.read_csv(url, header=None)


df.info()
print(df.describe())
print(df.corr())
print(df[0].value_counts())
# X, y = df.iloc[:, :-1], df.iloc[:, -1]
X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype('float32')
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, shuffle=True)


n_features = X.shape[1]

kf = KFold(n_splits=10)

scores = []

for train_ix, test_ix in kf.split(X, y):
    X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], X[train_ix], X[test_ix]

    model = Sequential(
        [
            Dense(20, input_shape=(n_features,)),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    )

    model.compile(optimizer='adam', loss='binary_crossentropy')

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    y_pred = model.predict(X_test)
    y_classes = np.argmax(y_pred, axis=1)

    score = accuracy_score(y_test, y_test.round())

    print(f'Accuracy: {score}')

    plt.title('Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.grid()
    plt.legend()
    plt.show()

    scores.append(score)

print(f'Average Accuracy for Cross Validation: {sum(scores) / len(scores)}')
