#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: Akihiro Inui
"""
# Import libraries/modules
import os
import pandas as pd
import numpy as np
import time
from backend.src.common.config_reader import ConfigReader
from backend.src.preprocess.audio_preprocess import AudioPreProcess
from backend.src.feature_extraction.fft import FFT
from backend.src.feature_extraction.zerocrossing import zerocrossing
from backend.src.feature_extraction.mfcc import MFCC
from backend.src.feature_extraction.centroid import centroid
from backend.src.feature_extraction.rolloff import rolloff
from backend.src.feature_extraction.rms import rms
from backend.src.feature_extraction.flux import Flux
from backend.src.feature_extraction.osc import OSC
from backend.src.feature_extraction.low_energy import low_energy
from backend.src.feature_extraction.modulation_spectrum_feature import MSF
from backend.src.utils.stats_tool import get_mean, get_std
from backend.src.utils.file_utils import FileUtil
from backend.src.data_process.data_process import DataProcess


class AudioFeatureExtraction:
    """
    Audio feature extraction to audio files
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
        self.mod_fft_size = self.cfg.mod_fft_size

        # Initialize pre-processing
        self.APP = AudioPreProcess(self.frame_time, self.overlap_rate, self.window_type)

        # Feature selection
        self.short_feature_selection_dict = self.cfg.section_reader("short_feature_selection")
        self.long_feature_selection_dict = self.cfg.section_reader("long_feature_selection")
        self.short_feature_list = self.__init_short_feature_select()
        self.long_feature_list = self.__init_long_feature_select()

        # Initialize feature extraction classes
        self.mfcc = MFCC(self.cfg.mfcc_coeff, self.sampling_rate, self.fft_size, self.cfg.mfcc_total_filters)
        self.flux = Flux(self.sampling_rate)
        self.osc = OSC(self.cfg.osc_param, self.sampling_rate, self.fft_size)
        self.msf = MSF(self.cfg.omsc_param, self.sampling_rate, self.fft_size, self.mod_fft_size)

    def __init_short_feature_select(self) -> list:
        """
        Extract setting for short-term feature extraction from config file
        :return list of features to extract
        """
        short_feature_list = []
        for short_feature, switch in self.short_feature_selection_dict.items():
            if switch == "True":
                short_feature_list.append(short_feature)
        return short_feature_list

    def __init_long_feature_select(self) -> list:
        """
        Extract setting for long-term feature extraction from config file
        :return list of features to extract
        """
        long_feature_list = []
        for short_feature, switch in self.long_feature_selection_dict.items():
            if switch == "True":
                long_feature_list.append(short_feature)
        return long_feature_list

    # Pre-processing
    def pre_processing(self, audio_file: str) -> tuple:
        """
        Pre-processing to audio file
        :param  audio_file: name of audio file
        :return tuple of pre-processed audio signal
        """
        return self.APP.apply(audio_file)

    # Feature extraction to one frame
    def extract_short_frame(self, framed_audio: tuple):
        """
        Short-term feature extraction to one frame
        :param  framed_audio: tuple of framed audio data from audio file
        :return power_spectrum: power_spectrum from short-term frame
        :return dictionary of extracted features from framed audio data
                {key: name of feature, value: tuple of features from all frames}
        """
        # Store extracted features into a dictionary (key:name of feature, value: list of extracted features in frames)
        feature_dict = {}
        # Apply feature extraction to a framed audio and store into a dictionary
        # TODO: Use While True iterate over selected feature and use apply for each feature extraction method
        for short_feature in self.short_feature_list:
            # Apply feature extraction
            spectrum = FFT.fft(framed_audio, self.fft_size)
            power_spectrum = FFT.power_fft(framed_audio, self.fft_size)

            if short_feature == "zcr":
                feature_dict[short_feature] = zerocrossing(framed_audio)
            if short_feature == "mfcc":
                feature_dict[short_feature] = self.mfcc.main(spectrum)
            if short_feature == "rms":
                feature_dict[short_feature] = rms(framed_audio)
            if short_feature == "centroid":
                feature_dict[short_feature] = centroid(power_spectrum, self.fft_size, self.sampling_rate)
            if short_feature == "rolloff":
                feature_dict[short_feature] = rolloff(power_spectrum, self.cfg.rolloff_param)
            if short_feature == "flux":
                feature_dict[short_feature] = self.flux.main(power_spectrum)
            if short_feature == "osc":
                feature_dict[short_feature] = self.osc.main(power_spectrum)
            if short_feature == "mel_spectrogram":
                # Mel-spectrum needs to be stored to be converted later
                feature_dict[short_feature] = self.mfcc.mel_spectrum(spectrum)
        return power_spectrum, feature_dict

        # Feature extraction to one frame
    def extract_long_frame(self, long_frame_audio: list, long_frame_spectrum: list) -> dict:
        """
        Long-term feature extraction to one frame
        :param  long_frame_audio: list of audio data from short-term frame
        :param  long_frame_spectrum: list of spectrum from short-term frame
        :return dictionary of extracted features from framed audio data
                {key: name of feature, value: tuple of features from all frames}
        """
        # Store extracted features into a dictionary (key:name of feature, value: list of extracted features in frames)
        feature_dict = {}

        # Apply feature extraction to a framed audio and store into a dictionary
        for long_feature in self.long_feature_list:
            if long_feature == "low_energy":
                feature_dict[long_feature] = low_energy(long_frame_audio)
            if long_feature == "omsc":
                feature_dict[long_feature] = self.msf.omsc(long_frame_spectrum, self.mod_fft_size)
            if long_feature == "msfm":
                feature_dict[long_feature] = self.msf.msfm(long_frame_spectrum, self.mod_fft_size)
            if long_feature == "mscm":
                feature_dict[long_feature] = self.msf.mscm(long_frame_spectrum, self.mod_fft_size)
        return feature_dict

    def extract_file(self, input_audio_file: str):
        """
        Feature extraction to one audio file
        :param  input_audio_file: name of the audio file
        :return dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        :return file_short_feature_list: short-term features across whole audio file stored into list
        """
        # Pre-processing to audio file
        processed_audio = self.pre_processing(input_audio_file)

        # Apply feature extraction to all frames and store into dictionary
        feature_dict = {}
        frame_number = 0
        long_frame_audio = []
        long_frame_power_spectrum = []

        # Store whole short-term features in list
        file_short_feature_list = []
        for short_frame_audio in processed_audio:
            # Extract short-term features
            short_frame_power_spectrum, short_feature_dict = self.extract_short_frame(short_frame_audio)
            # Create a feature vector and append
            file_short_feature_list.append(DataProcess.flatten_list(list(short_feature_dict.values())))

            # Store short-term features in dictionary
            for short_feature_type in self.short_feature_list:
                feature_dict.setdefault(short_feature_type, []).append(short_feature_dict[short_feature_type])

            # Extract long-term features
            if frame_number == self.cfg.long_frame_length:
                long_feature_dict = self.extract_long_frame(long_frame_audio, long_frame_power_spectrum)
            # Store long-term features in dictionary
                for long_feature in self.long_feature_list:
                    feature_dict.setdefault(long_feature, []).append(long_feature_dict[long_feature])

            # Reset for long-term feature
                frame_number = 0
                long_frame_audio = []
                long_frame_power_spectrum = []

            # Update short frame stack
            frame_number += 1
            long_frame_audio.append(short_frame_audio)
            long_frame_power_spectrum.append(short_frame_power_spectrum)

        return feature_dict, file_short_feature_list

    def extract_directory(self, input_directory: str, stats_type: str):
        """
        Feature extraction to a folder which contains audio files
        :param  input_directory: folder name which has audio files
        :param  stats_type: type of statistics
        :return dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        :return directory_short_feature_list: short-term features extracted from all audio files from one directory
        """
        # Extract file names in the input directory
        file_names = FileUtil.get_file_names(input_directory)

        # Extract features from audio files in a directory
        file_feature_stat_dict = {}
        start = time.time()

        # Extract each audio file
        for count, audio_file in enumerate(file_names):
            # Extract features from one audio file
            frame_extracted_feature, file_short_feature_list = self.extract_file(os.path.join(input_directory, audio_file))

            # Append short-term feature to 3D array
            if count == 0:
                directory_3d_feature = np.array(file_short_feature_list)
                print(directory_3d_feature.shape)
            else:
                print(directory_3d_feature.shape)
                directory_3d_feature = np.dstack((directory_3d_feature, np.array(file_short_feature_list)))

            # Take stats across frames
            file_feature_stat_dict[audio_file] = self.get_feature_stats(frame_extracted_feature, stats_type)

        end = time.time()
        print("Extracted {0} with {1} \n".format(input_directory, end - start))

        return file_feature_stat_dict, directory_3d_feature

    def extract_dataset(self, dataset_path: str, stats_type: str):
        """
        Feature extraction to dataset
        Extract time series feature as 2D pandas dataframe and 3D numpy array, as well as label vector as list
        :param  dataset_path: path to dataset
        :param  stats_type: type of statistics for 2D feature
        :return all_2d_dataframe: 2D feature pandas dataframe across all frames
        :return all_3d_array: 3D feature numpy array across all frames
        :return label_list: list of numerical label vector
        """
        # Get folder names under data set path
        directory_names = FileUtil.get_folder_names(dataset_path, sort=True)

        # Get file names and store them into a dictionary
        directory_files_dict = {}
        for directory in directory_names:
            directory_files_dict[directory] = FileUtil.get_file_names(os.path.join(dataset_path, directory))

        # Extract all features and store them into list
        all_2d_dataframe = pd.DataFrame()
        dir_num = 0
        label_list = []
        for directory, audio_files in directory_files_dict.items():
            # Apply feature extraction to a directory
            file_feature_stat_dict, class_3d_feature = self.extract_directory(os.path.join(dataset_path, directory), stats_type)

            # Convert dictionary to data frame
            class_2d_dataframe = DataProcess.dict2dataframe(file_feature_stat_dict, segment_feature=True)

            # Add label to 2D feature data frame
            class_2d_dataframe_with_label = DataProcess.add_label(class_2d_dataframe, directory)

            # Combine 2D feature data frame
            all_2d_dataframe = all_2d_dataframe.append(class_2d_dataframe_with_label)

            # Append 3D arrays
            if dir_num == 0:
                all_3d_array = class_3d_feature
            else:
                all_3d_array = np.dstack((all_3d_array, class_3d_feature))

            # Make label as list
            a = [dir_num] * len(audio_files)
            label_list.extend(a)
            dir_num += 1

        # Transpose 3D array
        all_3d_array = all_3d_array.T

        return all_2d_dataframe, all_3d_array, label_list

    @staticmethod
    def get_feature_stats(feature_frame_dict: dict, stat_type: str) -> dict:
        """
        # Store statistics from features into dictionary
        :param  feature_frame_dict:dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        :param  stat_type: type of statistics
        :return feature_stat_dict: features from one audio file with statistics
                {key: name of feature, value: array or single value}
        """
        # For each feature, compute statistical operation
        feature_stat_dict = {}

        for feature, frame in feature_frame_dict.items():
            if stat_type == "mean":
                feature_stat_dict[feature] = get_mean(feature_frame_dict[feature], "r")
            elif stat_type == "std":
                feature_stat_dict[feature] = get_std(feature_frame_dict[feature], "r")
        return feature_stat_dict
