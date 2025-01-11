#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:58:25 2022

@author: alexander
"""


import re

from pandas import DataFrame


def trim_string(string: str, fill: str = ' ') -> str:
    return fill.join(filter(bool, re.split(r'\W', string))).lower()


def standardize_data(df: DataFrame) -> DataFrame:
    df.columns = map(lambda _: trim_string(_, fill='_'), df.columns)
    return df.dropna(axis=0, how='all').dropna(axis=1, how='all')
