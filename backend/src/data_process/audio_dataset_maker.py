#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 06

@author: Akihiro Inui
"""
# Import libraries/modules
import os
import librosa
from backend.src.utils.file_utils import FileUtil
from backend.src.utils.audio_util import AudioUtil
from backend.src.common.config_reader import ConfigReader


class AudioReaderError(Exception):
    pass


class AudioDatasetMaker:
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
        self.cfg = ConfigReader(setting_file)
        self.sampling_rate = self.cfg.sampling_rate
        self.audio_length = self.cfg.audio_length
        self.normalize = self.cfg.normalize

    def read_audio_file(self, input_audio_file_path: str, sampling_rate: int) -> tuple:
        """
        Read audio file as tuple
        :param input_audio_file_path: path to input audio file
        :param sampling_rate: sampling rate
        :return audio signal in tuple
        """

        try:
            audio_data = librosa.core.load(FileUtil.replace_backslash(input_audio_file_path), sampling_rate)
        except:
            raise AudioReaderError("Could not load audio file")
        return audio_data

    def clip_audio_signal(self, input_audio_signal: tuple, audio_length_sample: int):
        """
        Clip audio signal
        :param input_audio_signal: audio signal in tuple
        :param audio_length_sample: target audio length in sample
        :return clipped audio signal in numpy array
        """
        assert len(input_audio_signal[0]) > audio_length_sample, "Input audio file length is too short"
        return input_audio_signal[0][:audio_length_sample]

    def process_audio_file(self, input_audio_file_path: str, audio_length_second: int, sampling_rate: int):
        """
        Run process to one audio file
        :param input_audio_file_path: Path to input audio file
        :param audio_length_second: target audio length in seconds
        :param sampling_rate: sampling rate
        :return Processed audio in numpy array
        """
        # Read audio file
        audio_signal = self.read_audio_file(input_audio_file_path, sampling_rate)
        # Clip audio file
        processed_audio = self.clip_audio_signal(audio_signal, audio_length_second * sampling_rate)
        # TODO: more process??
        return processed_audio

    def save_audio_file(self, input_audio_signal, output_audio_file_path: str, sampling_rate: int,
                        normalize: bool) -> None:
        """
        Save audio signal into wav file
        :param input_audio_signal: input audio signal in numpy array
        :param output_audio_file_path: Path to output audio file
        :param sampling_rate: sampling rate
        :param normalize: normalize (True/False)
        """
        # Save audio file
        librosa.output.write_wav(output_audio_file_path, input_audio_signal, sampling_rate, normalize)

    def process_directory(self, input_directory: str, output_directory: str):
        """
        Run process to the input directory where audio files exist and save processed audio files
        :param input_directory: Path to input audio file
        :param output_directory: Output directory
        """
        # Make directory if it does not exist
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        # Extract file names in the input directory
        file_names = FileUtil.get_file_names(input_directory)

        # Apply process to audio files in the input directory
        for audio_file in file_names:
            # Process one audio file
            processed_audio = self.process_audio_file(os.path.join(input_directory, audio_file), self.audio_length,
                                                      self.sampling_rate)
            # Save it
            self.save_audio_file(processed_audio, os.path.join(output_directory, audio_file), self.sampling_rate,
                                 self.normalize)

    def process_dataset(self, input_dataset: str, output_dataset: str):
        """
        Run process to the dataset where directories exist
        :param input_dataset: Path to input dataset
        :param output_dataset: Output dataset
        """
        # Make directory if it does not exist
        if not os.path.isdir(output_dataset):
            os.mkdir(output_dataset)

        # Extract file names in the input directory
        folder_names = FileUtil.get_folder_names(input_dataset)

        # Apply process to audio files in the input directory
        for folder in folder_names:
            # Status
            print("Processing {}".format(folder))
            # Process one audio file
            self.process_directory(os.path.join(input_dataset, folder), os.path.join(output_dataset, folder))
            # Status
            print("Processed {}".format(folder))


if __name__ == "__main__":
    # File path
    setting_file = "../../../config/master_config.ini"
    input_dataset_path = "../../../data"
    output_dataset_path = "../../../processed_music_data"

    # Instantiate class
    ADM = AudioDatasetMaker(setting_file)

    # Process dataset
    ADM.process_dataset(input_dataset_path, output_dataset_path)
