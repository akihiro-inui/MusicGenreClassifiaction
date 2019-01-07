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
# Define Spectral Rolloff
#==============================================================================

def rolloff (param,X1):
    
    # Initialize energy and FFT number
    Energy = 0
    Count = 0
    
    # Find Count which has energy below param*TotalEnergy 
    TotalEnergy = sum(X1**2)
    
    # Find Count which has energy below param*TotalEnergy 
    while Energy <= param*TotalEnergy and Count < len(X1):
        Energy = X1[Count]**2 + Energy
        Count += 1
        
    # Adjust the order
    r = Count - 1
    
    # Normalise Spectral Rolloff
    r = r/len(X1)
    
    return r