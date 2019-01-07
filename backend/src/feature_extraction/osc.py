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
# 1. Define Spectral Rolloff
#==============================================================================

def osc (Fs,X1,fftSize,alpha):
    
    # Indicate frequency points to create bins
    Subband_points = [0,100,200,400,800,1600,3200,6400,12800,Fs/2]
    
    # FFT bins within half fftSize
    SubFFTbins = np.round(np.divide(Subband_points,Fs/2)*fftSize/2) 

    # Set the first value to 1
    SubFFTbins[0] = 1

    # Create empty matrices for peak, valley and sum for each band
    peak = np.zeros(len(Subband_points)-1)
    valley = np.zeros(len(Subband_points)-1)
    Xsum = np.zeros(len(Subband_points)-1)

    # Take peaks and valleys from all FFT frames
    for b in range (0,len(Subband_points)-1):   
        Xframe = X1[int(SubFFTbins[b]):int(SubFFTbins[b+1])]     # Take out FFT frame
        Xsmall2big = np.sort(Xframe)                             # Sort values from small to big
        Xbig2small = np.flipud(Xsmall2big)                       # Sort values from big to small
        N = int(np.round(alpha*len(Xframe)))                     # Take values up to N in each frame
        peak[b] = math.log10((1/N)*sum(Xbig2small[0:N]))         # Calculate peak from each frame
        valley[b] = math.log10((1/N)*sum(Xsmall2big[0:N]))       # Calculate valley from each frame
        Xsum[b] = sum(Xframe)                                    # Sum of power spectrum from each sub-band
        Xsum.transpose

    # Take difference (ignore the first value)
    sc = peak[:] - valley[:]

    # Cobmine features
    o = np.hstack((valley[:],sc))
    
    return o,Xsum