#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: Akihiro Inui
"""
# Import libraries/modules
import os
import pandas as pd
import time
from src.common.config_reader import ConfigReader
from src.preprocess.audio_preprocess import AudioPreProcess
from src.feature_extraction.fft import FFT
from src.feature_extraction.zerocrossing import zerocrossing
from src.feature_extraction.mfcc import MFCC
from src.feature_extraction.centroid import centroid
from src.feature_extraction.rolloff import rolloff
from src.feature_extraction.rms import rms
from src.feature_extraction.flux import Flux
from src.feature_extraction.osc import OSC
from src.utils.stats_tool import get_mean, get_std
from src.utils.file_utils import FileUtil
from src.data_process.data_process import DataProcess


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

        # Initialize feature extraction classes
        self.mfcc = MFCC(self.cfg.mfcc_coeff, self.sampling_rate, self.fft_size, self.cfg.mfcc_total_filters)
        self.flux = Flux(self.sampling_rate)
        self.osc = OSC(self.cfg.osc_param, self.sampling_rate, self.fft_size)

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
    def pre_processing(self, audio_file: str) -> tuple:
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
            # Apply feature extraction
            spectrum = FFT.fft(framed_audio, self.fft_size)
            power_spectrum = FFT.power_fft(framed_audio, self.fft_size)
            if feature == "zcr":
                feature_dict[feature] = zerocrossing(framed_audio)
            if feature == "mfcc":
                feature_dict[feature] = self.mfcc.main(spectrum)
            if feature == "rms":
                feature_dict[feature] = rms(framed_audio)
            if feature == "centroid":
                feature_dict[feature] = centroid(power_spectrum, self.fft_size, self.sampling_rate)
            if feature == "rolloff":
                feature_dict[feature] = rolloff(power_spectrum, self.cfg.rolloff_param)
            if feature == "flux":
                feature_dict[feature] = self.flux.main(power_spectrum)
            if feature == "osc":
                osc, fft_bin_sum = self.osc.main(power_spectrum)
                feature_dict[feature] = osc
            if feature == "mel_spectrogram":
                # Mel-spectrum needs to be stored to be converted later
                feature_dict[feature] = (self.mfcc.mel_spectrum(spectrum))
        return feature_dict

    def extract_file(self, input_audio_file: str) -> dict:
        """
        Feature extraction to one audio file
        :param  input_audio_file: name of the audio file
        :return dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        """
        # Pre-processing to audio file
        processed_audio = self.pre_processing(input_audio_file)

        # Apply feature extraction to all frames and store into dictionary
        feature_frame_dict = {}
        for frame in range(0, len(processed_audio[0]) - 1):
            frame_feature_dict = self.extract_frame(processed_audio[frame])
            for feature in self.feature_list:
                feature_frame_dict.setdefault(feature, []).append(frame_feature_dict[feature])
        return feature_frame_dict

    def extract_directory(self, input_directory: str, stats_type: str):
        """
        Feature extraction to a folder which contains audio files
        :param  input_directory: folder name which has audio files
        :param  stats_type: type of statistics
        :return dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        """
        # Extract file names in the input directory
        file_names = FileUtil.get_file_names(input_directory)

        # Extract features from audio files in a directory
        file_feature_stat_dict = {}
        start = time.time()
        for audio_file in file_names:
            # Extract features from one audio file
            frame_extracted_feature = self.extract_file(os.path.join(input_directory, audio_file))
            # Append mel-spectrum
            # mel_spectrogram = frame_extracted_feature["mel_spectrogram"]
            # Save mel-spectrogram image
            # plt.imshow(np.array(mel_spectrogram).T[:], aspect='auto')
            # plt.gca().invert_yaxis()
            # image_filename = "{}.png".format(audio_file.split(".wav")[0])
            # plt.savefig()
            # Remove mel_spectrogram from feature dictionary and calculate statistics over all the frames
            # del frame_extracted_feature["mel_spectrogram"]
            file_feature_stat_dict[audio_file] = self.get_feature_stats(frame_extracted_feature, stats_type)
        end = time.time()

        print("Extracted {0} with {1} \n".format(input_directory, end - start))

        return file_feature_stat_dict

    def extract_dataset(self, dataset_path: str, stats_type: str):
        # Get folder names under data set path
        directory_names = FileUtil.get_folder_names(dataset_path)

        # Get file names and store them into a dictionary
        directory_files_dict = {}
        for directory in directory_names:
            directory_files_dict[directory] = FileUtil.get_file_names(os.path.join(dataset_path, directory))

        # Extract all features and store them into list
        final_dataframe = pd.DataFrame()
        for directory, audio_files in directory_files_dict.items():
            # Apply feature extraction to a directory
            file_feature_stat_dict = self.extract_directory(os.path.join(dataset_path, directory), stats_type)

            # Convert dictionary to data frame
            class_dataframe = DataProcess.dict2dataframe(file_feature_stat_dict, segment_feature=True)

            # Add label to data frame
            class_dataframe_with_label = DataProcess.add_label(class_dataframe, directory)

            # Combine data frames
            final_dataframe = final_dataframe.append(class_dataframe_with_label)

        return final_dataframe

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

#    def short_term_feature_extraction(self, framed_audio):
        # :TODO Add 6 low-level feature extraction
