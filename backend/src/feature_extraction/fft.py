#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import math
import numpy as np


class FFT:
    """
    Different type of ffts to input audio signal
    """

    @staticmethod
    def fft(audio_signal: tuple or list, fft_size: int) -> list:
        """
        Standard fft with normalization
        :param  audio_signal: tuple/list of framed audio data from audio file
        :param  fft_size: size of fft
        :return spectrum in list
        """
        return abs(np.fft.fft(audio_signal, n=fft_size)) / math.sqrt(fft_size * len(audio_signal))

    @staticmethod
    def power_fft(audio_signal: tuple or list, fft_size: int) -> list:
        """
        Standard fft with normalization
        :param  audio_signal: tuple/list of framed audio data from audio file
        :param  fft_size: size of fft
        :return power spectrum in list
        """
        spectrum = abs(np.fft.fft(audio_signal, n=fft_size)) / math.sqrt(fft_size * len(audio_signal))
        # Truncate second half of fft
        return spectrum[0:int(fft_size/2)]

    @staticmethod
    def fft2long(framed_audio_list: list, fft_size: int):
        """
        Apply fft to stack of audio frames
        :param  framed_audio_list: list of framed audio data (tuple) from audio file
        :param  fft_size: size of fft
        """
        # Apply fft to each row to get modulation spectrum
        return abs(np.fft.fft(framed_audio_list, n=fft_size, axis=1))
