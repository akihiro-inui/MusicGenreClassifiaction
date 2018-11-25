#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import numpy as np

def centroid (X1,fft_size,Fs):
    
    # Calculate frequency bins
    k = (Fs/fft_size)*np.arange(0, int(fft_size/2))

    # Calculate Spectral Centroid
    c = sum(k*X1)/sum(X1)

    # Output normalized spectral centroid
    return c/(Fs/2)
