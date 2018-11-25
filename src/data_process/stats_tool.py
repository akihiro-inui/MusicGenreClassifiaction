#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: Akihiro Inui
This script contains tools for statistical analysis
"""
import numpy as np


def get_mean(input_data, axis:str = None):
    if np.shape(input_data)[0] == 1:
        return np.mean(input_data)
    else:
        if axis == "c":
            print(input_data)
            return np.mean(input_data, axis=0)
        elif axis == "r":
            print(input_data)
            return np.mean(input_data, axis=1)
