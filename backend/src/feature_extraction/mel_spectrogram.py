#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat 30 Aug 2019
@author: Akihiro Inui
"""

# Import libraries/modules
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def mel_spectrogram(input_audio_file: str, fft_size: int, num_mels: int, normalize: bool = True):
    """
    Extract Mel-spectrogram from one audio file.
    :param input_audio_file: Input audio file path
    :param fft_size: FFT size
    :param num_mels: Number of mels
    :return:
    """
    # Read audio file
    audio_data, sampling_rate = librosa.load(input_audio_file)

    # Extract mel-spectrogram from entire audio file
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate,
                                                     n_fft=fft_size, n_mels=num_mels)
    # Convert to log scale
    # mel_spectrogram = np.log(mel_spectrogram + 1e-9)

    if normalize is True:
        # Normalization
        mel_spectrogram = librosa.util.normalize(mel_spectrogram)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    return mel_spectrogram
