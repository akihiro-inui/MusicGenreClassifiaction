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
    Diffrent type of fft algorithms.
    """

    @staticmethod
    def fft(audio_signal: tuple or list, fft_size: int) -> list:
        return abs(np.fft.fft(audio_signal, n=fft_size)) / math.sqrt(fft_size * len(audio_signal))

    @staticmethod
    def power_fft(audio_signal: tuple or list, fft_size: int) -> list:
        spectrum = abs(np.fft.fft(audio_signal, n=fft_size)) / math.sqrt(fft_size * len(audio_signal))
        return spectrum[0:int(fft_size/2)]
