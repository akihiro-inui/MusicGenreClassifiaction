#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

#==============================================================================
# datacleaner.py
# Program author: Akihiro Inui
# Remove NaN or/and Inf from input data
#==============================================================================


#==============================================================================
# 0. Import libraries
#==============================================================================
import math
import numpy as np
#==============================================================================
# 1. Define data cleaning
#==============================================================================
# RawData: Data to be cleaned
# mode: 0:Exclude Inf and NaN, 1:Exclude NaN 2: Exclude Inf
# rc: 0: Exclude rows with errors, 1: Exclude colmuns with errors

def datacleaner(RawData,mode,rc):
    
    # Case of vector input data
    if RawData.ndim == 1:
        NaNData = np.zeros(len(RawData))
        InfData = NaNData
        
        if mode == 0:
            for i in range(0,len(RawData)):
                NaNData[i] = math.isnan(RawData[i])
                InfData[i] = math.isinf(RawData[i])
        BrokenData = NaNData + InfData
        ValidDataIdx = np.nonzero(BrokenData == 0)[0]
        ValidData = RawData[ValidDataIdx]
        
        if mode == 1:
            for i in range(0,len(RawData)):
                NaNData[i] = math.isnan(RawData[i])
        BrokenData = NaNData
        ValidDataIdx = np.nonzero(BrokenData == 0)[0]
        ValidData = RawData[ValidDataIdx]
        
        if mode == 2:
            for i in range(0,len(InfData)):
                InfData[i] = math.isnan(RawData[i])
        BrokenData = InfData
        ValidDataIdx = np.nonzero(BrokenData == 0)[0]
        ValidData = RawData[ValidDataIdx]
        
    # Case of matrix input data
    if RawData.ndim ==  2:
        
        if mode == 0:
        # Check data for each row
            for r in range(0,((RawData.shape)[0])):
                for c in range(0,((RawData.shape)[1])):
                    # Replace Inf as NaN
                    if math.isinf(RawData[r,c]) == 1:
                        RawData[r,c] = np.nan
            # Exclude colmun/row including NaN
            if rc == 0: 
                ValidData = RawData[~np.isnan(RawData).any(axis=1)]             # Exclude Row
            if rc == 1:
                ValidData = np.ma.compress_cols(np.ma.masked_invalid(RawData))  # Exclude Column
                
        if mode == 1:
            if rc == 0:
                ValidData = RawData[~np.isnan(RawData).any(axis=1)]             # Exclude Row
            if rc == 1:
                ValidData = np.ma.compress_cols(np.ma.masked_invalid(RawData))  # Exclude Column
    
        if mode == 2:
            if rc == 0:
                ValidData = RawData[~np.isinf(RawData).any(axis=1)]             # Exclude Row
            if rc == 1:
                ValidData = np.ma.compress_cols(np.ma.masked_invalid(RawData))  # Exclude Column
    
    return ValidData