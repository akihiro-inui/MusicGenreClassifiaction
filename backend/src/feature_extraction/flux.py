#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: Akihiro Inui
"""
import math


class Flux:
    """
    Spectral Flux
    """
    def __init__(self, sampling_rate: int):
        """
        Init for spectral flux
        :param  sampling_rate: Sampling rate
        """
        self.sampling_rate = sampling_rate
        # Initialize power spectrum
        self.previous_power_spectrum = 0

    def main(self, input_power_spectrum: list) -> float:
        """
        Main function for spectral flux
        :param  input_power_spectrum: power spectrum from current frame in list
        :return Spectral Flux
        """

        # Update power spectrum
        flux = math.sqrt((sum(input_power_spectrum-self.previous_power_spectrum)**2))/(self.sampling_rate/2)
        self.previous_power_spectrum = input_power_spectrum
        return flux
