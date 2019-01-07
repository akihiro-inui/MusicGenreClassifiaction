#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

#==============================================================================
# zerocrossing.py
# Program author: Akihiro Inui
# Compute Zero-crossing Rate
#==============================================================================

#==============================================================================
# 0. Import libraries
#==============================================================================
import numpy as np
#==============================================================================
# 1. Define Zero-Crossing Rate
#==============================================================================

def zerocrossing (xw):
    
    # Size of windowed signal
    wsize = len(xw)
    
    # Slided signal
    xw2 = np.zeros(wsize)
    xw2[1:] = xw[0:-1]
    
    # Compute Zero-crossing Rate
    z = (1/(2*wsize)) * sum(abs(np.sign(xw)-np.sign(xw2)))
    
    return z