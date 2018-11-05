#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 2018

@author: Akihiro Inui
"""

import math
import numpy as np
from utils.audio_util import AudioUtil


class AudioPreProcess:

    # Initialization
    def __init__(self, input_audio_file: str):
        """
        Constractor for audio pre-processing
        :param input_audio_file: file path to the input audio file
        """
        self.filename = input_audio_file

    def read_audio_file(self):
        """
        Read audio file
        :return Dictionary of audio file parameters {key:name of information, value: parameters}
        """
        if AudioUtil.is_wav_file(self.filename):
            return AudioUtil.audio_read(self.filename)

    def framing(self, frame_time: float, overlap_rate: float) -> tuple:
        """
        Segmentation forthe input audio file with the given frame length
        :param frame_time: time for framing in ms (e.g. 46.44)
        :param overlap_rate: overlap between frames
        :return tuple of the segmented audio file
        """
        # Read audio file
        audio_extracted = self.read_audio_file()

        # Samples in one frame
        one_frame_samples = math.floor((frame_time / 1000) * audio_extracted["sampling_rate"])
        self.one_frame_samples = one_frame_samples

        # Calculate samples in frame step
        step_samples = math.floor(one_frame_samples * overlap_rate)

        # Calculate number of frames
        frame_number = int(math.floor((len(audio_extracted["data"]) - step_samples) / step_samples))
        print("Frame length is {0} with {1}% overlap".format(one_frame_samples, int(100 * overlap_rate)))
        print("{} frames from the input audio file".format(frame_number))

        # Store framed audio input into tuple
        framed_audio_tuple = {}
        for frame in range(frame_number):
            framed_audio_tuple[frame] = audio_extracted["data"][frame * step_samples:(frame + 1) * step_samples]
        return framed_audio_tuple

    def windowing(self, input_audio: list, window_type: str, window_length: int) -> list:
        """
        Apply window to the audio signal
        :param input_audio: input audio to be windowed
        :param window_type: type of window
        :param window_length: length of window
        :return list of windowed audio signal
        """
        if window_type is "hamm":
            windowed_signal = input_audio * np.hamming(window_length)
        return windowed_signal
