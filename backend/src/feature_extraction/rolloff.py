#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: Akihiro Inui
"""


def rolloff(input_power_spectrum: list, param: float=0.85) -> float:
    """
    Spectral Rolloff
    :param  input_power_spectrum: power spectrum in list
    :param  param: threadshold for roll off
    :return Spectral Rolloff
    """
    assert (param <= 0 or param >= 1) is False, "parameter must be between 0 and 1"
    # Initialize energy and FFT number
    energy = 0
    count = 0
    
    # Calculate total energy
    total_energy = sum(input_power_spectrum[:]**2)
    
    # Find Count which has energy below param*total_energy
    while energy <= param*total_energy and count < len(input_power_spectrum):
        energy = pow(input_power_spectrum[count], 2) + energy
        count += 1
    # Normalise Spectral Rolloff
    return count/len(input_power_spectrum)
