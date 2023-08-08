#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:58:35 2022

@author: alexander
"""


from zipfile import ZipFile

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

with ZipFile('../data/external/titanic.zip').open('train.csv') as f:
    df = pd.read_csv(f)

df = df[["Survived", "Sex", "Age", "SibSp", "Parch", "Fare"]]
df.dropna(inplace=True)
le = LabelEncoder()
le.fit(df["Sex"])
df["Sex"] = le.transform(df["Sex"])
scaler = MinMaxScaler()
scaler.fit(df)
print(scaler.data_max_)
print(scaler.transform(df).sum())

scaler = StandardScaler()
scaler.fit(df)
print(scaler.transform(df).sum())
