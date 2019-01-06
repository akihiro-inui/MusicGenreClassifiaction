#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: Akihiro Inui
"""
# Import libraries/modules
from common.config_reader import ConfigReader
from preprocess.audio_preprocess import AudioPreProcess
from feature_extraction.fft import FFT
from feature_extraction.mfcc import MFCC
from feature_extraction.centroid import centroid
from utils.stats_tool import get_mean


class AudioFeatureExtraction:
    """
    Audio feature extraction to one audio file
    Supported features: mfcc, spectral centroid,
    """

    # Initialization
    def __init__(self, setting_file: str):
        """
        Initialization for parameters and classes
        :param setting_file: config file
        """
        # Load parameters from config file
        self.cfg = ConfigReader(setting_file)
        self.sampling_rate = self.cfg.sampling_rate
        self.frame_time = self.cfg.frame_time
        self.overlap_rate = self.cfg.overlap_rate
        self.window_type = self.cfg.window_type
        self.fft_size = self.cfg.fft_size

        # Initialize pre-processing
        self.APP = AudioPreProcess(self.frame_time, self.overlap_rate, self.window_type)

        # Feature selection
        self.feature_selection_dict = self.cfg.section_reader("feature_selection")
        self.feature_list = self.__init_feature_select()

        # Initialize mfcc
        self.mfcc = MFCC(self.cfg.mfcc_coeff, self.sampling_rate, self.fft_size, self.cfg.mfcc_total_filters)

    def __init_feature_select(self) -> list:
        """
        Extract setting for feature extraction from config file
        :return list of features to extract
        """
        feature_list = []
        for feature, switch in self.feature_selection_dict.items():
            if switch == "True":
                feature_list.append(feature)
        return feature_list

    # Pre-processing
    def preprocessing(self, audio_file: str) -> tuple:
        """
        Pre-processing to audio file
        :param  audio_file: name of audio file
        :return tuple of pre-processed audio signal
        """
        return self.APP.apply(audio_file)

    # Feature extraction to one frame
    def extract_frame(self, framed_audio: tuple) -> dict:
        """
        Feature extraction to one frame
        :param  framed_audio: tuple of framed audio data from audio file
        :return dictionary of extracted features from framed audio data
                {key: name of feature, value: tuple of features from all frames}
        """
        # Store extracted features into a dictionary (key:name of feature, value: list of extracted features in frames)
        feature_dict = {}

        # Apply feature extraction to a framed audio and store into a dictionary
        for feature in self.feature_list:
            # Apply feature extraction only the feature type is set as True
            spectrum = FFT.fft(framed_audio, self.fft_size)
            power_spectrum = FFT.power_fft(framed_audio, self.fft_size)
            if feature == "mfcc":
                feature_dict[feature] = self.mfcc.main(FFT.fft(framed_audio, self.fft_size))
            if feature == "centroid":
                feature_dict[feature] = centroid(power_spectrum, self.fft_size, self.sampling_rate)
        return feature_dict

    # Feature extraction to one audio file
    def extract_file(self, audio_file: str) -> dict:
        """
        Feature extraction to one audio file
        :param  audio_file: name of the audio file
        :return dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        """
        # Pre-processing to audio file
        processed_audio = self.preprocessing(audio_file)

        # Apply feature extraction to all frames and store into dictionary
        feature_frame_dict = {}
        for frame in range(0, len(processed_audio[0]) - 1):
            frame_feature_dict = self.extract_frame(processed_audio[frame])
            for feature in self.feature_list:
                feature_frame_dict.setdefault(feature, []).append(frame_feature_dict[feature])
        return feature_frame_dict

    @staticmethod
    def get_feature_stats(feature_frame_dict: dict, stat_type: str) -> dict:
        """
        # Store statistics from features into dictionary
        :param  feature_frame_dict:dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        :return feature_stat_dict: features from one audio file with statistics
                {key: name of feature, value: array or single value}
        """
        # For each feature, compute statistical operation
        feature_stat_dict = {}
        for feature, frame in feature_frame_dict.items():
            if stat_type == "mean":
                feature_stat_dict[feature] = get_mean(feature_frame_dict[feature], "r")
        return feature_stat_dict
