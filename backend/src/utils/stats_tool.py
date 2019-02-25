#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018
@author: Akihiro Inui
This script contains functions for statistical analysis
"""
import numpy as np


def get_mean(input_data, axis: str = "r"):
    """
    Mean value of input data
    :param  input_data: tuple/list of framed audio data from audio file
    :param  axis: "r" for mean of rows, "c" for mean of columns
    :return mean value
    """
    # Case of vector input
    if np.shape(input_data)[0] == 1:
        return np.mean(input_data)
    # Case of matrix input
    else:
        if axis == "r":
            return np.nanmean(input_data, axis=0)
        elif axis == "c":
            return np.nanmean(input_data, axis=1)


def get_std(input_data, axis: str = "r"):
    """
    Standard fft with normalization
    :param  input_data: tuple/list of framed audio data from audio file
    :param  axis: "r" for std of rows, "c" for std of columns
    :return standard deviation value
    """
    # Case of vector input
    if np.shape(input_data)[0] == 1:
        return np.std(input_data)
    # Case of matrix input
    else:
        if axis == "r":
            return np.nanstd(input_data, axis=0)
        elif axis == "c":
            return np.nanstd(input_data, axis=1)


def geo_mean(num_list):
    return np.exp(np.log(num_list).sum() / len(np.log(num_list)))
