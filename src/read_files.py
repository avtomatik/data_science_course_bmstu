#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:59:40 2022

@author: green-machine
"""

import datetime

import pandas as pd


def parse_dates(string: str) -> datetime.date:
    date_parsed = list(map(int, string.split('-M'))) + [1]
    return datetime.date(*date_parsed)


def show_case(**kwargs):
    df = pd.read_csv(**kwargs)

    print(df.info())


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
        'filepath_or_buffer': '../data/ebw_data.csv',
        # 'names': ,
    },
    {
        'filepath_or_buffer': '../data/praktika_regressiya_1_mnk.csv',
        # 'names': ,
    },
    {
        'filepath_or_buffer': '../data/trade.txt',
        'sep': '\t',
        # 'names': ,
        'index_col': 0,
        'date_parser': parse_dates,
        'encoding': 'cp1251',
    },
    {
        'filepath_or_buffer': '../data/vote.txt',
        'sep': '\t',
        # 'names': ,
        'index_col': 0,
        'encoding': 'cp1251',
    },
)


for kwargs in kwargs_collection:
    show_case(**kwargs)
