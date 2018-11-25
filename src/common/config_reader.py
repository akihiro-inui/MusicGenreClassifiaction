#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018
@author: Akihiro Inui
"""

import configparser
from utils.file_utils import FileUtils


class ConfigReader:
    """
    Reading configuration file data. Define module specific configuration in different functions.
    """
    def __init__(self, i_config_path: str="config/master_config.ini"):
        """
        Read common configuration data from configuration file
        """
        cfg = configparser.ConfigParser()
        self.cfg = cfg
        config_file = i_config_path
        assert FileUtils.is_valid_file(config_file), config_file + " is not a valid configuration file!"
        cfg.read(config_file)

        # Dataset directory
        self.dataset_directory = cfg.get('main_config', 'dataset_directory')

        # Read module specific config reader
        self.__init_audio_preprocess(cfg)
        self.__init_feature_selection(cfg)
        self.__init_feature_extraction(cfg)

    def __init_audio_preprocess(self, cfg):
        # Parameters for pre-process
        self.sampling_rate = float(cfg.get('preprocess', 'sampling_rate'))
        self.frame_time = float(cfg.get('preprocess', 'frame_time'))
        self.overlap_rate = float(cfg.get('preprocess', 'overlap_rate'))
        self.window_type = str(cfg.get('preprocess', 'window_type'))

    def __init_feature_selection(self, cfg):
        # Parameters for feature selection
        self.fft = bool(cfg.get('feature_selection', 'fft'))

    def __init_feature_extraction(self, cfg):
        # Parameters for feature extraction
        self.fft_size = int(cfg.get('feature_extraction', 'fft_size'))
        self.mfcc_coeff = int(cfg.get('feature_extraction', 'mfcc_coeff'))
        self.mfcc_total_filters = int(cfg.get('feature_extraction', 'mfcc_total_filters'))

    def section_reader(self,section_name: str) -> dict:
        # Read parameters from a given section
        param_dict = {}
        options = self.cfg.options(section_name)
        for option in options:
            try:
                param_dict[option] = self.cfg.get(section_name, option)
                if param_dict[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                param_dict[option] = None
        return param_dict

if __name__ == '__main__':
    """
    usage of config reader
    """
    configuration_reader = ConfigReader()
    print(configuration_reader.input_directory)
    print(configuration_reader.output_directory)
    print(configuration_reader.corpus_file_name)
    print(configuration_reader.corpus_file)
    print(configuration_reader.sub_document_directory)
    print(configuration_reader.temp_dir)
