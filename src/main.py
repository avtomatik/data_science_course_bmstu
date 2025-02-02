#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 22:05:19 2022

@author: alexander
"""

from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

DIR = '../data/external'


# df = pd.read_csv(
#     Path(DIR).joinpath('iris.csv'),
#     # header=None,
#     # names=['value']
#     # encoding='unicode_escape'
# )

# # one_hot_encoder = OneHotEncoder()
# # one_hot_encoder.fit(np.array(df.variety.values).reshape(-1, 1))
# # targets_trans = one_hot_encoder.transform(
# #     np.array(df.variety.values).reshape(-1, 1))
# #
# MAP = {title: _ for _, title in enumerate(df.variety.unique(), start=1)}
# df['class'] = df.variety.map(MAP)
# X = [df.iloc[_, :4].tolist() for _ in range(df.shape[0])]
# y = df.iloc[:, -1].tolist()
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
# print(neigh.predict([[5, 5, 2, 1]]))
# print(neigh.predict_proba([[8, 4, 5, 3]]))


df = pd.read_csv(
    Path(DIR).joinpath('creditaustralia.csv'),
    # header=None,
    # names=['value']
    # encoding='unicode_escape'
)
X = [df.iloc[_, :-1].tolist() for _ in range(df.shape[0])]
y = df.iloc[:, -1].tolist()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[5, 5, 2, 1]]))
print(neigh.predict_proba([[8, 4, 5, 3]]))
