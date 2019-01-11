#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: Akihiro Inui
"""

import math


def rms(input_windowed_signal: list or tuple) -> float:
    """
    Root mean square energy
    :param  input_windowed_signal: input audio signal after windowing
    :return Root mean square energy
    """
    return math.sqrt(1/len(input_windowed_signal)*sum(input_windowed_signal[:]**2))
