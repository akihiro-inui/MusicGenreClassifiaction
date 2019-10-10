#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018
@author: Akihiro Inui
"""
# Import libraries/modules
import os
import time
import numpy as np
from tqdm import tqdm
from backend.src.common.config_reader import ConfigReader
from backend.src.preprocess.audio_preprocess import AudioPreProcess
from backend.src.feature_extraction.mel_spectrogram import mel_spectrogram
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
        self.window_size = int(self.sampling_rate*0.001*self.frame_time)
        self.hop_size = int(self.window_size*self.cfg.overlap_rate)

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
        # Apply FFT
        spectrum = FFT.fft(framed_audio, self.fft_size)
        power_spectrum = FFT.power_fft(framed_audio, self.fft_size)

        # Apply feature extraction to a framed audio and store into a dictionary
        feature_dict = {}
        for short_feature in self.short_feature_list:
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
        return power_spectrum, feature_dict

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

    def extract_entire_audio(self, input_audio_file: str):
        """
        Read audio file and extract Mel-spectrogram
        :param input_audio_file: Input audio file
        :return: Mel-spectrogram: Mel-spectrogram(currently) in numpy 2D array
        """
        # Read audio file and extract mel-spectrogram from entire audio signal
        return mel_spectrogram(input_audio_file, self.fft_size, self.cfg.num_mels, normalize=True)

    def extract_file(self, input_audio_file: str):
        """
        Feature extraction to one audio file
        :param  input_audio_file: name of the audio file
        :return dictionary of extracted features from audio file
                {key: name of feature, value: list of array(number of frames)}
        """
        # Prepare a dictionary to store extracted feature
        feature_dict = {}

        # Pre-processing to audio file
        processed_audio = self.pre_processing(input_audio_file)

        # Extract Mel-spectrogram from the entire audio
        feature_dict['mel_spectrogram'] = self.extract_entire_audio(input_audio_file)

        # Apply feature extraction to all frames and store into dictionary
        short_frame_number = 0
        long_frame_audio = []
        long_frame_power_spectrum = []

        # Store whole short-term features in list
        for short_frame_audio in processed_audio:
            # Extract short-term features
            short_frame_power_spectrum, short_feature_dict = self.extract_short_frame(short_frame_audio)

            # Store short-term features in dictionary
            for short_feature_type in self.short_feature_list:
                feature_dict.setdefault(short_feature_type, []).append(short_feature_dict[short_feature_type])

            # Extract long-term features when the number of short frames reach to a certain number
            if short_frame_number == self.cfg.long_frame_length:
                long_feature_dict = self.extract_long_frame(long_frame_audio, long_frame_power_spectrum)
                # Store long-term features in dictionary
                for long_feature in self.long_feature_list:
                    feature_dict.setdefault(long_feature, []).append(long_feature_dict[long_feature])

                # Reset cached short-term feature
                short_frame_number = 0
                long_frame_audio = []
                long_frame_power_spectrum = []

            # Update short-term feature cache
            short_frame_number += 1
            long_frame_audio.append(short_frame_audio)
            long_frame_power_spectrum.append(short_frame_power_spectrum)

        return feature_dict

    def extract_directory(self, input_directory: str):
        """
        Feature extraction to a folder which contains audio files
        :param  input_directory: folder name which has audio files
        :return dictionary of extracted features from audio file
                {key: name of files, value: list of extracted features}
        """
        # Extract file names in the input directory
        file_names = FileUtil.get_file_names(input_directory)

        # Extract features from audio files in a directory
        # file_feature_stat_dict = {}
        file_feature_dict = {}
        start = time.time()

        # Extract each audio file
        for count, audio_file in tqdm(enumerate(file_names)):
            # Extract features from one audio file
            file_feature_dict[audio_file] = self.extract_file(os.path.join(input_directory, audio_file))

        print("Extracted {0} with {1} \n".format(input_directory, time.time() - start))
        return file_feature_dict

    def extract_dataset(self, dataset_path: str):
        """
        Feature extraction to dataset
        Extract time series feature as 2D pandas dataframe and 3D numpy array, as well as label vector as list
        :param  dataset_path: path to dataset
        :return directory_files_feature_dict: dictionary of extracted features from all audio files in dataset folder
        {key: name of directory, value: list of file names {key: file name, value: list of extracted features}}
        :return label_list: list of numerical label vector
        """
        # Make label
        label_list = self.make_label_from_directory(dataset_path)

        # Get file names and store them into a dictionary
        directory_files_dict = {}
        for directory in FileUtil.get_folder_names(dataset_path, sort=True):
            directory_files_dict[directory] = FileUtil.get_file_names(os.path.join(dataset_path, directory))

        # Extract all features and store them into list
        directory_files_feature_dict = {}
        for directory, audio_files in tqdm(directory_files_dict.items()):
            # Apply feature extraction to one directory
            directory_files_feature_dict[directory] = self.extract_directory(os.path.join(dataset_path, directory))

        return directory_files_feature_dict, label_list

    @staticmethod
    def dict2array(directory_files_feature_dict: dict):
        """
        Convert extracted feature to
        :param directory_files_feature_dict: dictionary of extracted features from all audio files in dataset folder
        {key: name of directory, value: list of file names {key: file name, value: list of extracted features}}
        :return: expert_feature_2d_array: 2D Numpy array of extracted feature using expert system
        :return: mel_spectrogram_3d_array: 3D Numpy array of extracted mel-spectrogram
        """
        # Initialization
        processed_file = 0
        expert_feature_vector = []

        # Process for each class
        for class_name, file_feature_dict in directory_files_feature_dict.items():
            # Process for each file
            for file_name, feature_value_dict in file_feature_dict.items():
                file_feature_vector = []
                # Process for each feature
                for feature_name, feature in feature_value_dict.items():
                    # Take stats across frames for expert system and append to list
                    if type(feature) is list:
                        file_feature_array = np.array(feature[:])
                        if file_feature_array.ndim == 1:
                            file_feature_vector.append(np.mean(file_feature_array))
                        else:
                            file_feature_vector.extend(np.mean(file_feature_array, axis=0))
                    # Append mel-spectrogram to 3D array
                    else:
                        if processed_file == 0:
                            mel_spectrogram_3d_array = np.dstack((np.empty(np.shape(feature), int), feature))
                            mel_spectrogram_3d_array = mel_spectrogram_3d_array[:, :, 1]
                        else:
                            mel_spectrogram_3d_array = np.dstack((mel_spectrogram_3d_array, feature))

                # Append expert system feature vector
                expert_feature_vector.append(file_feature_vector)
                processed_file += 1

        # Transpose 3D array
        mel_spectrogram_3d_array = mel_spectrogram_3d_array.T
        # Convert list to 2D numpy array
        expert_feature_2d_array = np.array(expert_feature_vector)

        return expert_feature_2d_array, mel_spectrogram_3d_array

    @staticmethod
    def make_label_from_directory(dataset_path: str):
        # Init parameter
        dir_num = 0
        label_list = []

        # Iterate over directories
        for directory in FileUtil.get_folder_names(dataset_path, sort=True):
            # Make label as list
            label_list.extend([dir_num] * len(FileUtil.get_file_names(os.path.join(dataset_path, directory))))
            dir_num += 1
        return label_list

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

        for feature_name, values in feature_frame_dict.items():
            print(feature_name)
            if type(values[0]) is not list and values[0].ndim >= 2:
                if stat_type == "mean":
                    feature_frame_dict[feature_name] = np.mean(values[:], axis=0) + 1e-8
                elif stat_type == "std":
                    feature_stat_dict[feature_name] = np.std(values[:], axis=0)
            else:
                if stat_type == "mean":
                    feature_frame_dict[feature_name] -= np.mean(feature_frame_dict[feature_name], axis=0) + 1e-8
                elif stat_type == "std":
                    feature_stat_dict[feature_name] = get_std(feature_frame_dict[feature_name], "r")
        return feature_frame_dict
