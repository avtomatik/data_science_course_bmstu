#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:59:40 2022

@author: green-machine
"""

import pandas as pd

from config import DATA_DIR


def show_case(**kwargs):
    df = pd.read_csv(**kwargs)
    df.info()
    print(df.describe())


kwargs_collection = (
    {
        'filepath_or_buffer': DATA_DIR.joinpath('airline_passengers.csv'),
        'index_col': 0,
        'parse_dates': [0],
    },
    {
        'filepath_or_buffer': DATA_DIR.joinpath('credit.txt'),
        'sep': '\t',
        'encoding': 'cp1251',
    },
    {
        'filepath_or_buffer': DATA_DIR.joinpath('praktika_regressiya_1_mnk.csv'),
    },
)


for kwargs in kwargs_collection:
    show_case(**kwargs)
