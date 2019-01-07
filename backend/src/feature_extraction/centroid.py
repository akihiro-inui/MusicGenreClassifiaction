#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018
@author: akihiro inui
"""

import numpy as np

def centroid (power_spectrum:list, fft_size:int, sampling_rate:int) -> float:
    """
    Spectral centroid
    :param  power_spectrum: power spectrum in list
    :param  fft_size: size of fft
    :param  sampling_rate: int
    :return spectral centroid
    """
    # Calculate frequency bins
    bins = (sampling_rate/fft_size)*np.arange(0, int(fft_size/2))

    # Calculate Spectral Centroid
    centroid = sum(bins*power_spectrum)/sum(power_spectrum)

    # Output normalized spectral centroid
    return centroid/(sampling_rate/2)
