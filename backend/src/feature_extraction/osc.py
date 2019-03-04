#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import numpy as np
import math


class OSC:

    def __init__(self, osc_param: int, sampling_rate: int, fft_size: int):
        """
        Octave-based spectral contrast
        :param  osc_param: parameter for OSC
        :param  sampling_rate: int
        :param  fft_size: size of fft
        """
        self.osc_param = osc_param
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.sub_fft_bins = self.__init_sub_band()

    def __init_sub_band(self) -> list:
        """
        Init to create sub-band
        :return subbands: fft sub-bands up to half of sampling rate
        """
        # Indicate frequency points to create bins
        subband_points = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, self.sampling_rate/2]

        # FFT bins within half fft size
        subbands = np.ceil(np.divide(subband_points, self.sampling_rate/2)*self.fft_size/2)
        # Set the second element as 1 to make each bins to have at least 2 elements
        subbands[1] = 1
        return subbands

    def main(self, input_power_spectrum: list):
        """
        Main function for Octave-based spectral contrast
        :param  input_power_spectrum: power spectrum from one short-term frame
        :return low energy
        """
        # Create empty matrices for peak, valley and sum for each band
        peak_array = []
        valley_array = []

        # Take peaks and valleys from all FFT frames
        for bin_num in range(1, len(self.sub_fft_bins)):
            # Take out FFT bin
            fft_bin = input_power_spectrum[int(self.sub_fft_bins[bin_num-1]):int(self.sub_fft_bins[bin_num])]
            # Sort values from small to big
            small2big = np.sort(fft_bin)
            # Sort values from big to small
            big2small = np.flip(small2big)
            # Take values up to N in each frame
            threshold = int(np.ceil(self.osc_param*len(fft_bin)))
            # Calculate peak from each frame
            peak_array.append(math.log10((1/threshold)*sum(big2small[:threshold])))
            # Calculate valley from each frame
            valley_array.append(math.log10((1/threshold)*sum(small2big[:threshold])))

        # Take difference except the first element which is the same value
        sc = np.subtract(peak_array[1:], valley_array[1:])

        # Combine features
        return list(np.concatenate([valley_array, sc]))

