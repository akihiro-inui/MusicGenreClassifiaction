#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018
utils for audio file process such as read/write, volume as such.
@author: Akihiro Inui
"""

from scipy.io.wavfile import read, write


class AudioUtil:
    @staticmethod
    def is_wav_file(input_audio_file: str):
        """
        Check the file is wav file
        :param input_audio_file: the name of the file to read in
        """
        assert input_audio_file.endswith(".wav") == True, "{0} is not wav file".format(input_audio_file)
        return True

    @staticmethod
    def audio_read(input_audio_file: str, normalize: bool = True) -> dict:
        """
        Read audio file
        :param input_audio_file: the name of the file to read in
        :param normalize: normalise audio data to between -1 to 1
        """
        try:
            # Read audio file
            fs, audio_data = read(input_audio_file)

            # Extract one channel (0:left, 1:right) if audio file is stereo
            if audio_data.ndim == 2:
                audio_data = audio_data[:, 0]

            # Normalize audio input
            if normalize:
                audio_data = audio_data / max(abs(audio_data[:]))

            # Return error if it is an invalid audio file
            assert len(audio_data) != 0, "{0} is not a valid audio file".format(audio_file)

        except AssertionError as err:
            print('AssertionError :', err)
        return dict({'file_name': input_audio_file, 'sampling_rate': fs, 'data': audio_data})
