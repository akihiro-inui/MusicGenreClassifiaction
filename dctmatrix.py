#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

#==============================================================================
# rolloff.py
# Program author: Akihiro Inui
# Compute Spectral Rolloff
#==============================================================================


#==============================================================================
# 0. Import libraries
#==============================================================================
import math
import numpy as np
#==============================================================================
# 1. Define DCT matrix
#==============================================================================

def dctmatrix (totalfilters,mfcccoeff):
    
    # Create an matrix (mfcccoeff * totalfilters)
    [cc,rr] = np.meshgrid(range(0,totalfilters), range(0,mfcccoeff))
    
    # Calculate DCT
    c = np.sqrt(2 / totalfilters) * np.cos(math.pi * (2*cc + 1) * rr / (2 * totalfilters))
    c[0,:] = c[0,:] / np.sqrt(2)                     
    
    return c