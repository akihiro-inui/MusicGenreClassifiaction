#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import numpy as np
import math


class MFCC:
    def __init__(self, mfcc_coeff: int, sampling_rate: int, fft_size: int, total_filters: int):
        self.total_filters = total_filters
        self.mfcc_coeff = mfcc_coeff
        self.fft_size = fft_size
        self.sampling_rate = sampling_rate
        self.__init_melfilter()
        self.__init_dct_matrix()

    def __init_dct_matrix(self):
        self.dct_matrix = self.dctmatrix(self.total_filters, self.mfcc_coeff)

    def __init_melfilter(self):
        self.mel_filter = self.melfilter(self.sampling_rate, self.total_filters)

    def melfilter(self, sampling_rate, total_filters):
        # Maximum frequency of filter (avoid aliasing)
        maxF = sampling_rate / 2

        # Maximal Mel-frequency
        maxMelF = 2595 * np.log10(1 + maxF / 700)

        # Scatter points in Mel-frequency scale
        melpoints = np.arange(0, (total_filters + 2)) / (total_filters + 1) * maxMelF

        # Convert points in normal frequency scale
        points = 700 * (10 ** (melpoints / 2595) - 1)

        # DTF bins within half fftSize
        DFTbins = np.round(points / maxF * (self.fft_size / 2))

        # Set the first value to 1
        DFTbins[0] = 1

        # Create an empty matrix to store filter
        MelFilter = np.zeros((total_filters, self.fft_size))

        # Create Triangle filters by each row
        for n in range(0, total_filters):
            low = int(DFTbins[n])  # Triangle start
            center = int(DFTbins[n + 1])  # Top of the Triangle
            high = int(DFTbins[n + 2])  # Triangle end

            UpSlope = center - low  # Number of DFT points in lower side of Triangle
            DownSlope = high - center  # Number of DFT points in upper side of Triangle

            # Create lower side slope
            MelFilter[n, range(low - 1, center)] = np.arange(0, UpSlope + 1) / UpSlope

            # Create upper side slope
            MelFilter[n, range(center - 1, high)] = np.flipud(np.arange(0, DownSlope + 1) / DownSlope)

        return MelFilter

    def dctmatrix(self, totalfilters, mfcccoeff):
        # Create an matrix (mfcccoeff * totalfilters)
        [cc, rr] = np.meshgrid(range(0, totalfilters), range(0, mfcccoeff))

        # Calculate DCT
        dct_matrix = np.sqrt(2 / totalfilters) * np.cos(math.pi * (2 * cc + 1) * rr / (2 * totalfilters))
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return dct_matrix

    def main(self, input_spectrum: list) -> list:
        # Apply Mel scale filter
        mel_fft = np.matmul(self.mel_filter, input_spectrum)

        # Log scale
        ear_mag = np.log10(mel_fft ** 2)

        # Apply DCT to cepstrum
        mfcc = self.dct_matrix.dot(ear_mag)

        return mfcc
