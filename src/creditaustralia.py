#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:23:52 2022

@author: alexander
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../data/external/Credit_Card_Applications.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
# neigh.predict([[1.1]])
# neigh.predict_proba([[0.9]])
