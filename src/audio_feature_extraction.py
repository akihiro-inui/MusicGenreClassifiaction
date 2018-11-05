#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: Akihiro Inui
"""

# Import libraries/modules
import os
import math
import numpy as np
from scipy.io.wavfile import read
from src.centroid import centroid
from src.dctmatrix import dctmatrix
from src.melfilter import melfilter
from osc import osc
from src.rolloff import rolloff
from src.zerocrossing import zerocrossing
from src.datacleaner import datacleaner
from utils.audio_util import AudioUtil

# Import variables from config file

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
windowsample = math.floor((windowtime / 1000) * Fs)

# Overlap for window
overlaptime = 0.5 * windowtime

# Window step
windowstep = math.floor(Fs * ((windowtime - overlaptime) / 1000))

# Melfilter
MelFilter = melfilter(Fs, fftsize, totalfilters)

# DCT Matrix
c = dctmatrix(totalfilters, mfcccoeff)

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

# ==============================================================================
# 2. Define instance
# ==============================================================================
class AudioFeatureExtraction:

    # Constructor
    def __init__(self, input_audio_file:str):
        self.filename = input_audio_file

    def read_audio_file(self):
        if AudioUtil.is_wav_file(self.filename) is True:
            self.audio_data_dictionary = AudioUtil.audio_read(self.filename)

    def

