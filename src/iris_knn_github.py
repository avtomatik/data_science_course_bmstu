#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 22:05:19 2022

@author: alexander
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

URL = 'https://huggingface.co/datasets/kestra/datasets/resolve/65375731f1504e094c1b1f5e8fbd531a252e2117/csv/iris.csv'

URL = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'


df = pd.read_csv(URL)

X = df.iloc[:, :-1].values
y_raw = df.variety.values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)


one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(y_encoded.reshape(-1, 1))
y = one_hot_encoder.transform(y_encoded.reshape(-1, 1)).toarray()

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)

print(classifier.predict([[5, 5, 2, 1]]))
print(classifier.predict_proba([[8, 4, 5, 3]]))
