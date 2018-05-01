#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

#==============================================================================
# melfilter.py
# Program author: Akihiro Inui
# Create Mel-Filter
#==============================================================================

#==============================================================================
# 0. Import library
#==============================================================================
import numpy as np
#==============================================================================
# 1. Define Mel-Filter
#==============================================================================

def melfilter (Fs,fftSize,totalfilters):
    # Maximum frequency of filter (avoid aliasing)
    maxF = Fs/2   

    # Maximal Mel-frequency 
    maxMelF = 2595*np.log10(1+maxF/700)   
    
    # Scatter points in Mel-frequency scale
    melpoints = np.arange(0,(totalfilters+2))/(totalfilters+1) * maxMelF
    
    # Convert points in normal frequency scale
    points = 700*(10**(melpoints/2595)-1)
    
    # DTF bins within half fftSize
    DFTbins = np.round(points/maxF*(fftSize/2)) 
    
    # Set the first value to 1
    DFTbins[0] = 1
    
    # Create an empty matrix to store filter
    MelFilter = np.zeros((totalfilters,fftSize))
    
    # Create Triangle filters by each row (for MFCC)
    for n in range (0,totalfilters):
        low = int(DFTbins[n])           # Triangle start
        center = int(DFTbins[n+1])      # Top of the Triangle
        high = int(DFTbins[n+2])        # Triangle end
        
        UpSlope = center-low       # Number of DFT points in lower side of Triangle
        DownSlope = high-center    # Number of DFT points in upper side of Triangle
        
        # Create lower side slope
        MelFilter[n,range(low-1,center)] = np.arange(0,UpSlope+1)/UpSlope       
        
        # Create upper side slope
        MelFilter[n,range(center-1,high)] = np.flipud(np.arange(0,DownSlope+1)/DownSlope)  
        
    return MelFilter

        