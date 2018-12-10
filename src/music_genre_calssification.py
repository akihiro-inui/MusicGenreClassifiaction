#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import os
import time
import numpy as np
import pandas as pd
from utils.file_utils import FileUtil
from classifier.classifier import Classifier
from common.config_reader import ConfigReader
from data_process.data_process import DataProcess
from sklearn.model_selection import train_test_split
from feature_extraction.audio_feature_extraction import AudioFeatureExtraction


class MusicGenreClassification:
    """
    # Content-based music genre classification
    # 1. Frame based feature extraction
    # 2. Data processing (Normalization, Factorize label)
    # 3. Make data set (Shuffle order, train/test separation)
    # 4. Train model
    # 5. Evaluate performance for n time iteration
    """

    def __init__(self, audio_feature_extraction, classifier, dataset_path: str, setting_file: str):
        """
        Init
        :param  audio_feature_extraction: audio feature extraction class
        :param  classifier:               classifier class
        :param  dataset_path:             path to data set
        :param  setting_file:             config file
        """
        self.AFE = audio_feature_extraction(setting_file)
        self.CLF = classifier(setting_file)
        self.dataset_path = dataset_path
        self.cfg = ConfigReader(setting_file)
        self.setting_file = setting_file

    def feature_extraction(self):
        """
        Feature extraction to data set
        :return final_dataframe:  extracted feature in pandas data frame
        """
        # Get folder names under data set path
        directory_names = FileUtil.get_folder_names(self.dataset_path)

        # Get file names and store them into a dictionary
        directory_files_dict = {}
        for directory in directory_names:
            directory_files_dict[directory] = os.listdir(os.path.join(self.dataset_path, directory))

        # Extract all features and store them into list
        final_dataframe = pd.DataFrame()
        for directory, audio_files in directory_files_dict.items():
            start = time.time()
            file_feature_stat_dict = {}
            # Extract all audio files in one directory
            for audio_file in audio_files:
                # Extract features from one audio file
                file_feature_stat_dict[audio_file] = self.AFE.get_feature_stats(
                    self.AFE.extract_file(os.path.join(self.dataset_path, directory, audio_file)), "mean")
            end = time.time()

            # Convert dictionary to data frame
            class_dataframe = DataProcess.dict2dataframe(file_feature_stat_dict, segment_feature=True)

            # Add label to data frame
            class_dataframe_with_label = DataProcess.add_label(class_dataframe, directory)

            # Combine data frames
            final_dataframe = final_dataframe.append(class_dataframe_with_label)

            print("Extracted {0} with {1} \n".format(directory, end - start))

        return final_dataframe

    def data_process(self, dataframe, label_name: str):
        """
        Apply data process to features
        :param  dataframe:            extracted feature in data frame
        :param  label_name:           name of label column in data frame
        :return processed_dataframe:  extracted feature in pandas data frame
        """
        # Make a copy of dataframe
        processed_dataframe = dataframe.copy()
        # Apply normalization to data frame
        processed_dataframe = DataProcess.normalize_dataframe(processed_dataframe, label_name)
        # Factorize label
        processed_dataframe = DataProcess.factorize_lebel(processed_dataframe, label_name)
        return processed_dataframe

    def make_dataset(self, dataframe, label_name: str, test_size: float):
        """
        Make dataset
        :param  dataframe:   extracted feature in data frame
        :param  label_name:  name of label column in data frame
        :param  test_size:   size of test data set
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Split data and label
        label, data = DataProcess.data_label_split(dataframe, label_name)
        # Train/Test separation
        train_data, test_data, train_label, test_label = train_test_split(data,
                                                                          label,
                                                                          test_size=test_size,
                                                                          shuffle=True)
        return train_data, test_data, train_label, test_label

    def classification(self, dataframe, label_name: str, model_file_name: str) -> float:
        """
        Classification to data set
        :param  dataframe:       extracted feature in data frame
        :param  label_name:      name of label column in data frame
        :param  model_file_name: name of model file to load/save
        :return prediction accuracy
        """
        # Make data set from extracted features
        train_data, test_data, train_label, test_label = self.make_dataset(dataframe, label_name, self.cfg.test_rate)
        # Train classifier
        model = self.CLF.training(train_data, train_label, model_file_name)
        # Make prediction
        return self.CLF.predict(model, test_data, test_label)

    def classification_with_iteration(self, dataframe, label_name: str, model_file_name: str):
        """
        Classification to data set with iteration
        :param  dataframe:   extracted feature in data frame
        :param  label_name:  name of label column in data frame
        :param  model_file_name: name of model file to load/save
        :return prediction accuracy
        """
        # Iteration
        accuracy_list = []
        for itr in range(self.cfg.iteration):
            # Make data set from extracted features
            train_data, test_data, train_label, test_label = self.make_dataset(dataframe, label_name, self.cfg.test_rate)
            # Train classifier
            model = self.CLF.training(train_data, train_label, model_file_name)
            # Make predictions
            accuracy_list.append(self.CLF.predict(model, test_data, test_label))
        return np.average(accuracy_list)


def main():
    # File location
    setting_file = "../config/master_config.ini"
    model_file = "../model/mlp.h5"
    dataset_path = "../data"
    csv_file = "../feature/data.csv"

    # Instantiate mgc class
    MGC = MusicGenreClassification(AudioFeatureExtraction, Classifier, dataset_path, setting_file)

    # Apply feature extraction and write out csv file if it does not exist
    if not os.path.exists(csv_file):
        # Apply feature extraction to all audio files
        print("Start feature extraction")
        extracted_feature_dataframe = MGC.feature_extraction()
        FileUtil.dataframe2csv(extracted_feature_dataframe, csv_file)

    # Read data from csv file
    dataframe = FileUtil.csv2dataframe(csv_file)

    # Apply data process
    clean_data = MGC.data_process(dataframe, "category")

    # Classify
    accuracy = MGC.classification(clean_data, "category", model_file)
    print("Start prediction")
    print("Final accuracy is {0}".format(accuracy))


if __name__ == "__main__":
    main()
