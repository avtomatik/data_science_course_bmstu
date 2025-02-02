#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:59:40 2022

@author: green-machine
"""

import pandas as pd


def show_case(**kwargs):
    df = pd.read_csv(**kwargs)
    df.info()
    print(df.describe())


kwargs_collection = (
    {
        'filepath_or_buffer': '../data/airline_passengers.csv',
        # 'names': ,
        'index_col': 0,
        'parse_dates': [0],
    },
    {
        'filepath_or_buffer': '../data/credit.txt',
        'sep': '\t',
        # 'names': ,
        'encoding': 'cp1251',
    },
    {
        'filepath_or_buffer': '../data/praktika_regressiya_1_mnk.csv',
        # 'names': ,
    },
)


for kwargs in kwargs_collection:
    show_case(**kwargs)
