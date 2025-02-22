#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 22:05:19 2022

@author: alexander
"""


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

from config import DATA_DIR

df = pd.read_csv(
    DATA_DIR.joinpath('external').joinpath('iris.csv'),
    # header=None,
    # names=['value']
    # encoding='unicode_escape'
)

one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(np.array(df.variety.values).reshape(-1, 1))
targets_trans = one_hot_encoder.transform(
    np.array(df.variety.values).reshape(-1, 1)
)

MAP = {title: _ for _, title in enumerate(df.variety.unique(), start=1)}

df['class'] = df.variety.map(MAP)

X = [df.iloc[_, :4].tolist() for _ in range(df.shape[0])]
y = df.iloc[:, -1].tolist()

solver = KNeighborsClassifier(n_neighbors=3)
solver.fit(X, y)

print(solver.predict([[5, 5, 2, 1]]))
print(solver.predict_proba([[8, 4, 5, 3]]))
