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
        """
        Mel Frequency Cepstral Coefficient
        :param  mfcc_coeff: number of  coefficients
        :param  sampling_rate: int
        :param  fft_size: size of fft
        :param  total_filters: number of mel filters
        """
        self.total_filters = total_filters
        self.mfcc_coeff = mfcc_coeff
        self.fft_size = fft_size
        self.sampling_rate = sampling_rate
        self.total_filters = total_filters
        self.__init_melfilter()
        self.__init_dct_matrix()

    def __init_dct_matrix(self):
        """
        Init DCT matrix
        """
        self.dct_matrix = self.dctmatrix(self.total_filters, self.mfcc_coeff)

    def __init_melfilter(self):
        """
        Init Mel filter
        """
        self.mel_filter = self.melfilter(self.sampling_rate, self.fft_size, self.total_filters)

    @staticmethod
    def melfilter(sampling_rate:int, fft_size:int, total_filters:int):
        """
        Mel filter
        :param  total_filters: number of mel filters
        :return  Mel filter in list
        """
        # Maximum frequency of filter (avoid aliasing)
        maxF = sampling_rate / 2

        # Maximal Mel-frequency
        maxMelF = 2595 * np.log10(1 + maxF / 700)

        # Scatter points in Mel-frequency scale
        melpoints = np.arange(0, (total_filters + 2)) / (total_filters + 1) * maxMelF

        # Convert points in normal frequency scale
        points = 700 * (10 ** (melpoints / 2595) - 1)

        # DTF bins within half fftSize
        DFTbins = np.ceil(points / maxF * (fft_size / 2))

        # Set the first value to 0
        DFTbins[0] = 0

        # Create an empty matrix to store filter
        MelFilter = np.zeros((total_filters, fft_size))

        # Create Triangle filters by each row
        for n in range(0, total_filters):
            low = int(DFTbins[n])         # Triangle start
            center = int(DFTbins[n + 1])  # Top of the Triangle
            high = int(DFTbins[n + 2])    # Triangle end

            UpSlope = center - low        # Number of DFT points in lower side of Triangle
            DownSlope = high - center     # Number of DFT points in upper side of Triangle

            # Create lower side slope
            MelFilter[n, range(low - 1, center)] = np.arange(0, UpSlope + 1) / UpSlope

            # Create upper side slope
            MelFilter[n, range(center - 1, high)] = np.flipud(np.arange(0, DownSlope + 1) / DownSlope)

        return MelFilter

    @staticmethod
    def dctmatrix(total_filters: int, mfcc_coeff: int) -> bytearray:
        """
        DCT matrix
        :param  total_filters: number of mel filters
        :param  mfcc_coeff: number of  coefficients
        """
        # Create an matrix (mfcccoeff * totalfilters)
        [cc, rr] = np.meshgrid(range(0, total_filters), range(0, mfcc_coeff))

        # Calculate DCT
        dct_matrix = np.sqrt(2 / total_filters) * np.cos(math.pi * (2 * cc + 1) * rr / (2 * total_filters))
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return dct_matrix

    def main(self, input_spectrum: list):
        """
        Main function for Mel Frequency Cepstral Coefficient
        :param  input_spectrum: spectrum in list
        :return mfcc: mfccs in list
        :return mel_fft: mel-scaled fft
        """
        # Apply Mel scale filter
        mel_fft = np.matmul(self.mel_filter, input_spectrum)

        # Log scale
        ear_mag = np.log10(mel_fft ** 2)

        # Apply DCT to cepstrum
        return list(self.dct_matrix.dot(ear_mag))

    def mel_spectrum(self, input_spectrum: list):
        """
        Make Mel-spectrum from mel-scaled spectrum
        :param  input_spectrum: spectrum in list
        :return : mel-scaled spectrum
        """
        # Apply inverse FFT to mel-scaled spectrum and truncate half
        return list(abs(np.fft.ifft((np.matmul(self.mel_filter, input_spectrum)), self.fft_size))[:self.fft_size])

