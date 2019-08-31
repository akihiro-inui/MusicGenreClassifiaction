#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 2018
@author: Akihiro Inui
"""

import math
import numpy as np
from backend.src.utils.audio_util import AudioUtil


class AudioPreProcess:
    """
    Pre processing to audio file
    1. Framing
    2. Windowing (Options: Hamm window)
    """
    def __init__(self, frame_time: float, overlap_rate: float, window_type: str):
        """
        :param frame_time: time for framing in ms (e.g. 46.44)
        :param overlap_rate: overlap between frames
        :param window_type: type of window
        """
        self.frame_time = frame_time
        self.overlap_rate = overlap_rate
        self.window_type = window_type

    def read_audio_file(self, filename: str) -> dict:
        """
        Read audio file
        :param filename: name of input audio file
        :return Dictionary of audio file parameters {key:name of information, value: parameters}
        """
        if AudioUtil.is_wav_file(filename):
            return AudioUtil.audio_read(filename)

    @staticmethod
    def add_eps(audio_data: list) -> list:
        """
        Add eps to audio data
        :param audio_data: list of audio data
        :return list of audio data with eps
        """
        return audio_data + np.finfo(float).eps

    def framing(self, audio_dict: dict) -> tuple:
        """
        Segmentation for the input audio file with the given frame length
        :param audio_dict: audio data dictionary {key:name of information, value: parameters}
                                                 e.g. key: sampling_rate, value: int
        :return tuple of the segmented audio file
        """
        # Calculate samples in one frame
        one_frame_samples = math.floor((self.frame_time / 1000) * audio_dict["sampling_rate"])

        # Calculate samples in frame step
        step_samples = math.floor(one_frame_samples * self.overlap_rate)

        # Calculate number of frames
        frame_number = int(math.floor((len(audio_dict["data"]) - step_samples) / step_samples))

        # Add eps to raw audio data
        audio_dict["data"] = AudioPreProcess.add_eps(audio_dict["data"])

        # Store framed audio input into tuple
        framed_audio_list = []
        for frame in range(frame_number):
            framed_audio_list.append(audio_dict["data"][frame * step_samples:(frame + 1) * step_samples])
        return tuple(framed_audio_list)

    def windowing(self, framed_audio_tuple: tuple, window_type: str = "hamm") -> tuple:
        """
        Apply window to the audio signal
        :param framed_audio_tuple: input audio to be windowed
        :param window_type: type of window
        :return tuple of windowed audio signal
        """
        # Size of window
        window_size = len(framed_audio_tuple[0])

        # Apply window
        windowed_audio_list = []
        if window_type is "hamm":
            for frame in range(len(framed_audio_tuple)):
                windowed_audio_list.append(framed_audio_tuple[frame] * np.hamming(window_size))
        return tuple(windowed_audio_list)

    def apply(self, input_filename: str) -> tuple:
        """
        Apply pre-processing to input audio file.
        :param input_filename: input audio file
        :return tuple of pre-processed audio signal
        """
        return self.windowing(self.framing(self.read_audio_file(input_filename)))

