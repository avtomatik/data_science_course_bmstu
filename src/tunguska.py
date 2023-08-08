#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:47:18 2022

@author: alexander
"""

from zipfile import ZipFile

PATH = '../data/external/urovni_p_i_n_tunguski.zip'
with ZipFile(PATH) as archive:
    print(archive.namelist())
