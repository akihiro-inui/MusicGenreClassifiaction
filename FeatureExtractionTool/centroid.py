#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

#==============================================================================
# centroid.py
# Program author: Akihiro Inui
# Compute Spectral Centroid
#==============================================================================

#==============================================================================
# 0. Import libraries
#==============================================================================
import numpy as np
#==============================================================================
# 1. Define Spectral Centroid
#==============================================================================

def centroid (X1,fftSize,Fs):
    
    # Calculate frequency bins
    k = (Fs/fftSize)*np.arange(0,int(fftSize/2))

    # Calculate Spectral Centroid
    c = sum(k*X1)/sum(X1)

    # Normalise by Fs/2
    c = c/(Fs/2)
    
    return c