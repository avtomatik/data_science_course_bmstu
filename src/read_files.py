#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:59:40 2022

@author: green-machine
"""

import datetime
import os

import pandas as pd


def date_parser(string: str) -> datetime.date:
    date_parsed = [int(member) for member in string.split("-M")] + [1]
    return datetime.date(*date_parsed)


def show_case(**kwargs):
    df = pd.read_csv(**kwargs)
    print(df)
    print(df.info())


DIR = "/home/green-machine/data_science/bmstu/data"

kwargs_collection = (
    {
        "filepath_or_buffer": "airline_passengers.csv",
        # "names": ,
        "index_col": 0,
        "parse_dates": [0],
    },
    {
        "filepath_or_buffer": "credit.txt",
        "sep": "\t",
        # "names": ,
        "encoding": "cp1251",
    },
    {
        "filepath_or_buffer": "ebw_data.csv",
        # "names": ,
    },
    {
        "filepath_or_buffer": "praktika_regressiya_1_mnk.csv",
        # "names": ,
    },
    {
        "filepath_or_buffer": "trade.txt",
        "sep": "\t",
        # "names": ,
        "index_col": 0,
        "date_parser": date_parser,
        "encoding": "cp1251",
    },
    {
        "filepath_or_buffer": "vote.txt",
        "sep": "\t",
        # "names": ,
        "index_col": 0,
        "encoding": "cp1251",
    },
)

os.chdir(DIR)

for kwargs in kwargs_collection:
    show_case(**kwargs)
