#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: akihiro inui
"""

## USE IT LIKE BELOW ##########################################################
# TestData = FeatureExtraction("audio.wav")
# Features = TestData.Features()
# You need to change parameters depending on audio length(ideally to 30s length)
###############################################################################
#==============================================================================
# FeatureExtraction.py
# Program author: Akihiro Inui
# Feature Extraction to audio file
#==============================================================================
path = "[Your Working Directory]"
import os
os.chdir(path)
#==============================================================================
# 0. Import libraries
#==============================================================================
import math
import numpy as np
from scipy.io.wavfile import read
from ml.centroid import centroid
from ml.dctmatrix import dctmatrix
from ml.melfilter import melfilter
from osc import osc
from ml.rolloff import rolloff
from ml.zerocrossing import zerocrossing
from data_process.datacleaner import datacleaner
#==============================================================================
# 1. Preamble and Parameter setting
#==============================================================================
## Variable declaration for features

# Number of mfccs (+1 in order to truncate the first coefficient)
mfcccoeff = 15
mfcccoeff = mfcccoeff + 1

# Number of Octave based sub-bands
b = 8

# FFT size
fftsize = 4096

# Number of filters for mel-filter
totalfilters = 40

# Sampling frequency
Fs = 44100

# Length of analysis window
windowtime = 46.44

# Samples in one analysis window
windowsample = math.floor((windowtime/1000) * Fs)

# Overlap for window
overlaptime = 0.5 * windowtime

# Window step
windowstep = math.floor(Fs*((windowtime-overlaptime)/1000))

# Melfilter
MelFilter = melfilter(Fs,fftsize,totalfilters)

# DCT Matrix
c = dctmatrix(totalfilters,mfcccoeff)

# Number of Octave based sub-bands
b = 9

# Parameters for Octave based Spectral Contrast
alpha = 0.2

# Size of modulation spectrum
fftsize1 = 512   

# Number of sub-band for modulation spectrum                  
J = 8                             

# Silence Removal (0:Off,1:On)
Srem = 1

# Variable for modulation spectrum
valley = np.zeros(J)
contrast = np.zeros(J)
#==============================================================================
# 2. Define instance
#==============================================================================
class FeatureExtraction:

    # Constructor
    def __init__(self,file):
        
        # Load audio data
        fs, x = read(file)
    
        # Extract one channel (0:left, 1:right) if audio file is stereo
        if x.ndim == 2:
            x = x[:,0]
            
        # Normalize audio input
        audio = x/max(abs(x[:]))
        
        # Sampling frequency
        self.Fs = Fs
        
        # Window length and step
        self.windowtime =  windowtime
        self.windowsample = windowsample
        self.overlaptime = overlaptime
        self.windowstep = windowstep
        
        # Number of Analysis and Texture windows
        AFrameNum = int(np.floor((len(x)-windowstep)/windowstep))
        self.AFrameNum = AFrameNum
        TFrameNum = 10
        self.TFrameNum = TFrameNum
        
        # Storage for windowed input signal into matrix
        self.Audiomatrix = np.zeros((windowsample,AFrameNum))
    
        # Case of Silence Removal
        if Srem == 1:
            Valid = np.zeros(AFrameNum)
            for i in range(0,AFrameNum):
                StartAnalysis = i*windowstep                       # Start sample of frame
                EndAnalysis = StartAnalysis + windowsample         # End sample of frame
                self.Audiomatrix[:,i] = audio[StartAnalysis:EndAnalysis]        
                Nonzeros = np.size((abs(audio[StartAnalysis:EndAnalysis]) > 0.0001).nonzero())    # Number of zeros in a frame
                if Nonzeros > len(audio[StartAnalysis:EndAnalysis])/2:
                     Valid[i] = 1                                  # Valid only frames have samples more than half of frames
                
                # Extract only valid frames
                self.ValidFrames = np.flatnonzero(Valid)           # Number of valid frames
                self.AFrameNum = np.size(self.ValidFrames)         # New analysis frame number
                
        # Case of without Silence Removal
        else:
            for i in range(0,AFrameNum):
                StartAnalysis = i*windowstep                       # Start sample of frame
                EndAnalysis = StartAnalysis + windowsample         # End sample of frame    
                self.Audiomatrix[:,i] = audio[StartAnalysis:EndAnalysis]                  # Take out samples
                self.ValidFrames = np.arange(0,AFrameNum)          # Number of valid frames
                self.AFrameNum = np.size(self.ValidFrames)         # New analysis frame number

        # Number of analysis windows in a texture window
        self.t = np.floor(AFrameNum/TFrameNum)
        
        # Create an empty matrix to store MFCC
        self.MFCC = np.zeros((mfcccoeff, AFrameNum))
        
        # Create an empty matrix to store spectrogram
        self.Spectrogram = np.zeros((int(fftsize/2), AFrameNum))
        
        # Create an empty matrix to store Melspectrogram
        self.Melspectrogram = np.zeros((int(fftsize/2), AFrameNum))
        
        # Create an empty matrix to store Spectral centroid
        self.Centroid = np.zeros(AFrameNum)
        
        # Create an empty matrix to store Spectral Rolloff
        self.Rolloff = np.zeros(AFrameNum)
        
        # Create an empty matrix to store Spectral Flux
        self.Flux = np.zeros(AFrameNum)
        self.X1Prev = np.zeros(int(fftsize/2))
        
        # Create an empty matrix to store Zero-Crossing Rate
        self.ZCR = np.zeros(AFrameNum)
        
        # Create an empty matrix to store Octave-based Spectral Contrast
        self.OSC = np.zeros((b*2,AFrameNum))
        
        # Create an empty vector to store sum of power spectrum in sub-bands
        self.XSum = np.zeros((b,AFrameNum))
        
        # Create an empty matrix to store Root Mean Square Energy and Low Energy
        self.RMSAnalysis = np.zeros(AFrameNum)
    
        self.Low = np.zeros(TFrameNum)   
        self.OMSC = np.zeros((2*J,TFrameNum))
        self.MSFM = np.zeros((J,TFrameNum))
        self.MSCM = np.zeros((J,TFrameNum))
        
        # FFT bins equally distributed     
        self.MScalebinstep = (fftsize1/2/J)     
        
        # Create empty matrices to store long term features
        self.MFCC_Mean = np.zeros((mfcccoeff,TFrameNum))
        self.Centroid_Mean = np.zeros(TFrameNum)
        self.Rolloff_Mean = np.zeros(TFrameNum)
        self.Flux_Mean = np.zeros(TFrameNum)
        self.ZCR_Mean = np.zeros(TFrameNum)
        self.OSC_Mean = np.zeros((2*b,TFrameNum))
        
#==============================================================================
# 3. Methods for Features (loop for each analysis window)
#==============================================================================
    
    def features(self):
        
        # Import global variables
        global fftsize, windowsample
        
        # FFT to entire audio file
        Spectrum = np.absolute(np.fft.fft(self.audio[0:fftsize]))
        #freqList = np.fft.fftfreq(fftsize, d=1.0/Fs)    
        
        for n in range(0,self.AFrameNum):
            
            # Windowing
            xw = self.Audiomatrix[:,self.ValidFrames[n]] * np.hamming(self.windowsample) 
            
            # Spectrum
            X = abs(np.fft.fft(xw,n=fftsize))
            
#            # Avoid error
#            if np.count_nonzero(X) < len(X)/2:
#                self.RMSAnalysis[n] = np.nan
#                self.ZCR[n] = np.nan
#                self.Centroid[n] = np.nan
#                self.Rolloff[n] = np.nan
#                self.Flux[n] = np.nan
#                [self.OSC[:,n],self.XSum[:,n]] = np.ones(b*2) * np.nan, np.nan
#                self.M = np.ones(mfcccoeff) * np.nan
            #else:
            # Normalise
            X1 = X / math.sqrt(fftsize*windowsample)

            # Compute Root Mean Square Energy
            self.RMSAnalysis[n] = math.sqrt(1/len(xw)*sum(xw**2))
    
            # Compute Zero-Crossing Rate
            self.ZCR[n] = zerocrossing(xw)
            
            # Trancate half of spectrum
            X1 = X1[0:int(fftsize/2)]
                
            # Calculate Spectral Centroid
            self.Centroid[n] = centroid(X1,fftsize,Fs)
    
            # Calculate Spectral Rolloff
            self.Rolloff[n] = rolloff(0.89,X1)
        
            # Calculate Spectral Flux
            self.Flux[n] = math.sqrt((sum((X1 - self.X1Prev)**2))/(Fs/2))
            
            # Compute Octave-based Spectral Contrast
            [self.OSC[:,n],self.XSum[:,n]] = osc(Fs,X1,fftsize,alpha)
        
            #Store FFT result
            self.X1Prev = X1
            
            # Apply Mel scale filter
            Melfft = np.matmul(MelFilter,X)
        
            # Log scale
            earMag = np.log10(Melfft**2)
        
            # Apply DCT to cepstrum
            M = c.dot(earMag)
        
            #Store MFCC into matrix
            self.MFCC[:,n] = M
        
        # Remove useless data
        MFCC = datacleaner(self.MFCC,0,0)
        RMSAnalysis = datacleaner(self.RMSAnalysis,0,0)
        Centroid = datacleaner(self.Centroid,0,0)
        Rolloff = datacleaner(self.Rolloff,0,0)
        Flux = datacleaner(self.Flux,0,0)
        ZCR = datacleaner(self.ZCR,0,0)
        OSC = datacleaner(self.OSC,0,0)
        
        # For each texture window
        for l in range(0,self.TFrameNum):
            StartTexture = int(l*self.t)                      # Start point of texture window
            EndTexture = int(StartTexture + self.t)           # End point of texture window
            
            if EndTexture >= self.AFrameNum:
                EndTexture = self.AFrameNum-1           # End analysis window to avoid exceeding Analysis frame length
                
            # Average of RMS energy in texture window
            RMSAverage = np.mean(RMSAnalysis[StartTexture:EndTexture])
        
            # Store RMS energy from analysis window into texture window
            LowRMS = (RMSAverage > RMSAnalysis[StartTexture:EndTexture])
            
            if len(RMSAnalysis[StartTexture:EndTexture]) == 0:
                self.Low[l] = np.nan
            else:
            # Compute Low Energy
                self.Low[l] = (sum(LowRMS)/len(RMSAnalysis[StartTexture:EndTexture]))*100
        
            # Sum of power spectrum in Sub-band across texture window (8*32)
            T = np.arange(StartTexture,EndTexture)
            E = self.XSum[:,StartTexture:EndTexture]
            Epadded = np.hstack((E,np.zeros((b,fftsize1-len(T)))))   # Zero padding to make 512 length vector
            
            M = abs(np.fft.fft(Epadded,n=fftsize1,axis=1))           # Apply fft to each row to get modulation spectrum
            M = M[:,0:int(fftsize1/2)]                               # Truncate half
            
            for jj in range(0,J):
                Mb = M[jj,:]
                Start = int(jj*self.MScalebinstep)
                End = int(Start+self.MScalebinstep)
                Mframe = Mb[Start:End]                                         # Take out FFT frame
                peak = max(np.log10(Mframe[:]))                                # Calculate peaks from each frame
                minimum = min(np.log10(Mframe[:]))                             # Calculate valley from each frame
                valley[jj] = min(np.log10(Mframe[1:round(self.MScalebinstep/2)]))   # Search valley from first half frame     
                contrast[jj] = peak - minimum                                  # Calculate contrast
                
            # Combine features to create Octave-based Modulation Spectral Contrast
            self.OMSC[:,l] = np.hstack((contrast,valley))
            
            def geo_mean(iterable):
                a = np.log(iterable)
                return np.exp(a.sum()/len(a))
            
            for k in range(0,b-1):
                self.MSFM[k,l] = geo_mean(M[k,:])/np.mean(M[k,:])
                self.MSCM[k,l] = max(M[k,:])/np.mean(M[k,:])          
                
#==============================================================================
# 8. Compute long-term features (From Texture window)
#============================================================================== 
            self.MFCC_Mean[:,l] = np.mean(MFCC[:,StartTexture:EndTexture],axis=1)
            
        
            self.Centroid_Mean[l] = np.mean(Centroid[StartTexture:EndTexture])
            
        
            self.Rolloff_Mean[l] = np.mean(Rolloff[StartTexture:EndTexture])
            
        
            self.Flux_Mean[l] = np.mean(Flux[StartTexture:EndTexture])
            
        
            self.ZCR_Mean[l] = np.mean(ZCR[StartTexture:EndTexture])
            
        
            self.OSC_Mean[:,l] = np.mean(OSC[:,StartTexture:EndTexture],axis=1)
            
        
        MFCC = np.mean(self.MFCC_Mean,axis=1)
        MFCC = MFCC[1:]
        Centroid = np.mean(self.Centroid_Mean)
        Rolloff = np.mean(self.Rolloff_Mean)
        Flux = np.mean(self.Flux_Mean)
        ZCR = np.mean(self.ZCR_Mean)
        OSC = np.mean(self.OSC_Mean,axis=1)
        
        #self.Low[l] = np.mean(self.Low[StartTexture:EndTexture])
        Low = np.mean(self.Low,axis=0)
        
        #self.OMSC[l] = np.mean(self.OMSC[:,StartTexture:EndTexture])
        OMSC = np.mean(self.OMSC,axis=0)
        
        #self.MSFM[l] = np.mean(self.MSFM[:,StartTexture:EndTexture])
        MSFM = np.mean(self.MSFM,axis=0)
        
        #self.MSCM[l] = np.mean(self.MSCM[:,StartTexture:EndTexture])
        MSCM = np.mean(self.MSCM,axis=0)
            
            
        return MFCC, Centroid, Rolloff, Flux, ZCR, OSC, Low, OMSC, MSFM, MSCM

