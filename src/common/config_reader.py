#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018
@author: Akihiro Inui
"""

import configparser
from utils.file_utils import FileUtil


class ConfigReader:
    """
    Reading configuration file data. Define module specific configuration in different functions.
    """

    def __init__(self, i_config_path: str):
        """
        Read common configuration data from configuration file
        """
        cfg = configparser.ConfigParser()
        self.cfg = cfg
        config_file = i_config_path
        assert FileUtil.is_valid_file(config_file), config_file + " is not a valid configuration file!"
        cfg.read(config_file)

        # Dataset directory
        self.dataset_directory = cfg.get('main_config', 'dataset_directory')

        # Read module specific config reader
        self.__init_audio_preprocess(cfg)
        self.__init_feature_selection(cfg)
        self.__init_feature_extraction(cfg)
        self.__init_dataset(cfg)
        self.__init_classifier_selection(cfg)
        self.__init_classification(cfg)

    def __init_audio_preprocess(self, cfg):
        # Parameters for pre-process
        self.sampling_rate = float(cfg.get('preprocess', 'sampling_rate'))
        self.frame_time = float(cfg.get('preprocess', 'frame_time'))
        self.overlap_rate = float(cfg.get('preprocess', 'overlap_rate'))
        self.window_type = str(cfg.get('preprocess', 'window_type'))

    def __init_feature_selection(self, cfg):
        # Parameters for feature selection
        self.mfcc = bool(cfg.get('feature_selection', 'mfcc'))
        self.rms = bool(cfg.get('feature_selection', 'rms'))
        self.centroid = bool(cfg.get('feature_selection', 'centroid'))

    def __init_feature_extraction(self, cfg):
        # Parameters for feature extraction
        self.fft_size = int(cfg.get('feature_extraction', 'fft_size'))
        self.mfcc_coeff = int(cfg.get('feature_extraction', 'mfcc_coeff'))
        self.mfcc_total_filters = int(cfg.get('feature_extraction', 'mfcc_total_filters'))

    def __init_dataset(self, cfg):
        # Parameters for data set creation
        self.test_rate = float(cfg.get('dataset', 'test_rate'))
        self.label_name = str(cfg.get('dataset', 'label_name'))
        self.shuffle = bool(cfg.get('dataset', 'shuffle'))

    def __init_classifier_selection(self, cfg):
        # Parameters for classifier selection
        self.kNN = bool(cfg.get('classifier_selection', 'knn'))
        self.MLP = bool(cfg.get('classifier_selection', 'mlp'))

    def __init_classification(self, cfg):
        # Parameters for classification
        self.num_classes = int(cfg.get('classification', 'num_classes'))
        self.validation_rate = float(cfg.get('classification', 'validation_rate'))
        self.iteration = int(cfg.get('classification', 'iteration'))
        self.k = int(cfg.get('classification', 'k'))

    def section_reader(self, section_name: str) -> dict:
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
