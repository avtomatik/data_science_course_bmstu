#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:58:35 2022

@author: alexander
"""


from zipfile import ZipFile

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from config import DATA_DIR

# =============================================================================
# https://www.kaggle.com/competitions/titanic/data
# =============================================================================
archive_name = 'titanic.zip'

with ZipFile(
    DATA_DIR.joinpath('external').joinpath(archive_name)
).open('train.csv') as f:
    df = pd.read_csv(f, index_col=0)

df = df[['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
df.dropna(inplace=True)

label_encoder = LabelEncoder()
label_encoder.fit(df['Sex'])
df['Sex'] = label_encoder.transform(df['Sex'])

scaler = MinMaxScaler()
scaler.fit(df)
print(scaler.data_max_)
print(scaler.transform(df).sum())

scaler = StandardScaler()
scaler.fit(df)
print(scaler.transform(df).sum())
