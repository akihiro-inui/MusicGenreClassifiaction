#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018
@author: akihiro inui
"""

import numpy as np
import math


def low_energy(framed_audio_list) -> float:
    """
    Compute Low Energy
    :param  framed_audio_list: list of framed audio data (tuple) from audio file
    :return low energy
    """
    #
    rms = []
    for block in framed_audio_list:
        # Calculate RMS for each frame
        rms.append(math.sqrt(1/len(block)*sum(block**2)))
    rms_average = np.mean(rms)

    # Store low energy
    low_rms = (rms_average > rms)

    # Calculate low energy
    return sum(low_rms) / len(rms)


