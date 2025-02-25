#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:18:59 2025

@author: alexandermikhailov
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

X, y_encoded = load_iris(return_X_y=True)


one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(y_encoded.reshape(-1, 1))
y = one_hot_encoder.transform(y_encoded.reshape(-1, 1)).toarray()

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)

print(classifier.predict([[5, 5, 2, 1]]))
print(classifier.predict_proba([[8, 4, 5, 3]]))
