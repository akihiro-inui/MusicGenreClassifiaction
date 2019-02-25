#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 2019

@author: akihiro inui
"""

import numpy as np
from backend.src.feature_extraction.fft import FFT
from backend.src.utils.stats_tool import geo_mean


class MSF:
    """
    Modulation spectrum features
    """
    def __init__(self, omsc_param, sampling_rate: int, fft_size: int, mod_fft_size: int):
        """
        :param  sampling_rate: int
        :param  mod_fft_size: size of modulation fft
        """
        self.omsc_param = omsc_param
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.mod_fft_size = mod_fft_size
        self.binstep = mod_fft_size/2/omsc_param
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

    def calculate_sum(self, long_frame_power_spectrum: list):
        long_fft_bin_sum = []
        for short_frame_power_spectrum in long_frame_power_spectrum:
            # Create empty matrices for peak, valley and sum for each band
            fft_bin_sum = []
            # Take peaks and valleys from all FFT frames
            for bin_num in range(1, len(self.sub_fft_bins)):
                # Take out FFT bin
                fft_bin = short_frame_power_spectrum[int(self.sub_fft_bins[bin_num-1]):int(self.sub_fft_bins[bin_num])]
                # Sum of power spectrum from each sub-band
                fft_bin_sum.append(sum(fft_bin))
            long_fft_bin_sum.append(fft_bin_sum)
        return long_fft_bin_sum

    def omsc(self, long_frame_power_spectrum: list, mod_fft_size: int):
        """
        Octave-based Modulation spectral contrast
        :param  long_frame_power_spectrum: list of framed audio data (tuple) from audio file
        :param  mod_fft_size: fft size for modulation spectrum
        :return omsc
        """
        # Calculate sum of power spectrum for each short frame
        long_fft_bin_sum = np.asarray(self.calculate_sum(long_frame_power_spectrum)).T
        # Prepare zero vector to add later
        zero_vector = np.zeros(((mod_fft_size - long_fft_bin_sum.shape[1]), 1))
        # Create empty matrices for peak, valley and sum for each band
        valley_array = []
        contrast_array = []
        for bin_num in long_fft_bin_sum:
            # Zero padding
            zero_padded_bin_sum = np.append(bin_num, zero_vector)

            # Modulation spectrum
            mod_spectrum = FFT.power_fft(zero_padded_bin_sum, mod_fft_size)

            # Calculate minimum and peak
            peak = max(np.log10(mod_spectrum))
            minimum = min(np.log10(mod_spectrum))

            # Search valley from first half frame
            valley_array.append(min(np.log10(mod_spectrum[1:round(self.binstep / 2)])))

            # Calculate contrast
            contrast_array.append(peak - minimum)

        # Combine features
        return list(np.concatenate([valley_array, contrast_array]))

    def msfm(self, long_frame_power_spectrum: list, mod_fft_size: int):
        """
        Modulation Spectral Flatness Measure
        :param  long_frame_power_spectrum: list of framed audio data (tuple) from audio file
        :param  mod_fft_size: fft size for modulation spectrum
        :return msfm_array: msfm in list
        """
        # Calculate sum of power spectrum for each short frame
        long_fft_bin_sum = np.asarray(self.calculate_sum(long_frame_power_spectrum)).T
        # Prepare zero vector to add later
        zero_vector = np.zeros(((mod_fft_size - long_fft_bin_sum.shape[1]), 1))
        # Create empty matrices for peak, valley and sum for each band
        msfm_array = []
        for bin_num in long_fft_bin_sum:
            # Zero padding
            zero_padded_bin_sum = np.append(bin_num, zero_vector)

            # Modulation spectrum
            mod_spectrum = FFT.power_fft(zero_padded_bin_sum, mod_fft_size)

            # Compute msfm
            msfm_array.append(geo_mean(mod_spectrum) / np.mean(mod_spectrum))
        return msfm_array

    def mscm(self, long_frame_power_spectrum: list, mod_fft_size: int):
        """
        Modulation Spectral Crest Measure
        :param  long_frame_power_spectrum: list of framed audio data (tuple) from audio file
        :param  mod_fft_size: fft size for modulation spectrum
        :return mscm_array: mscm in list
        """
        # Calculate sum of power spectrum for each short frame
        long_fft_bin_sum = np.asarray(self.calculate_sum(long_frame_power_spectrum)).T
        # Prepare zero vector to add later
        zero_vector = np.zeros(((mod_fft_size - long_fft_bin_sum.shape[1]), 1))
        # Create empty matrices for peak, valley and sum for each band
        mscm_array = []
        for bin_num in long_fft_bin_sum:
            # Zero padding
            zero_padded_bin_sum = np.append(bin_num, zero_vector)

            # Modulation spectrum
            mod_spectrum = FFT.power_fft(zero_padded_bin_sum, mod_fft_size)

            # Compute mscm
            mscm_array.append(max(mod_spectrum) / np.mean(mod_spectrum))
        return mscm_array
