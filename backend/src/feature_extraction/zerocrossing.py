#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: Akihiro Inui
"""
import numpy as np


def zerocrossing(input_windowed_signal:tuple or list) -> float:
    """
    Zero Crossing Rate
    :param  input_windowed_signal: input audio signal after windowing
    :return zero crossing rate
    """
    # Size of windowed signal
    window_size = len(input_windowed_signal)
    
    # Slided signal
    xw2 = np.zeros(window_size)
    xw2[1:] = input_windowed_signal[0:-1]
    
    # Compute Zero-crossing Rate
    return (1/(2*window_size)) * sum(abs(np.sign(input_windowed_signal)-np.sign(xw2)))
