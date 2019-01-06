#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Christmas 2018

@author: Akihiro Inui
"""

import time
import pandas as pd
import numpy as np
import os
from utils.file_utils import FileUtil
from classifier.classifier_wrapper import Classifier
from common.config_reader import ConfigReader
from data_process.data_process import DataProcess
from feature_extraction.audio_feature_extraction import AudioFeatureExtraction


class MusicGenreClassification:
    """
    # Content-based music genre classification
    # 1. Frame based feature extraction
    # 2. Data processing (Normalization, Factorize label(str->num))
    # 3. Make data set (Shuffle order, train/test separation, write train.csv/test.csv in "feature" directory)
    # 4. Train model / Save model
    # 5. Make prediction
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
            directory_files_dict[directory] = FileUtil.get_file_names(os.path.join(self.dataset_path, directory))

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

    def make_dataset(self, dataframe, output_directory: str):
        """
        Make data set
        :param  dataframe:   extracted feature in data frame
        :param  output_directory: output directory to write out the train and test data
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Get time and make a new directory name
        directory_name_with_time = os.path.join(output_directory, FileUtil.get_time())

        train_data, test_data, train_label, test_label = DataProcess.make_dataset(dataframe, self.cfg.label_name,
                                                                                  self.cfg.test_rate, self.cfg.shuffle,
                                                                                  directory_name_with_time)
        return train_data, test_data, train_label, test_label

    def read_dataset(self, input_data_directory_with_date):
        """
        Read data set
        :param  input_data_directory_with_date: name of the directory where train and test data exist
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Read data set
        train_data, test_data, train_label, test_label = DataProcess.read_dataset(input_data_directory_with_date,
                                                                                  self.cfg.label_name)

        return train_data, test_data, train_label, test_label

    def data_process(self, dataframe):
        """
        Apply data process to features
        :param  dataframe:            extracted feature in data frame
        :param  label_name:           name of label column in data frame
        :return processed_dataframe:  extracted feature in pandas data frame
        """
        # Make a copy of dataframe
        processed_dataframe = dataframe.copy()
        # Apply normalization to data frame
        processed_dataframe = DataProcess.normalize_dataframe(processed_dataframe, self.cfg.label_name)
        # Factorize label
        processed_dataframe = DataProcess.factorize_lebel(processed_dataframe, self.cfg.label_name)
        return processed_dataframe

    def training(self, train_data, train_label, output_directory):
        """
        Train model and save it under output_directory
        :param  train_data:  train data
        :param  train_label: train label
        :return trained model
        """
        # Train classifier
        model = self.CLF.training(train_data, train_label)

        # Name the model with current time
        output_directory_with_time = os.path.join(output_directory, FileUtil.get_time())

        # Save mode
        self.CLF.save_model(model, output_directory_with_time)
        return model

    def test(self, model, test_data, test_label) -> float:
        """
        Make predictions to test data set
        :param  model:       trained model
        :param  test_data:   test data
        :param  test_label:  test label
        :return prediction accuracy
        """
        # Make prediction
        return self.CLF.test(model, test_data, test_label)

    def predict(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data
        :return prediction array with probability
        """
        return self.CLF.predict(model, target_data)


def main():
    # File location
    setting_file = "../config/master_config.ini"
    music_dataset_path = "../data"
    model_directory_path = "../model"
    output_data_directory = "../feature"
    feature_extraction = True
    training = True
    input_data_directory = "../feature/2019-01-03_23:13:27.829628"
    model_file = "../model/2019-01-03_23:21:07.913861/mlp.h5"
    dummy_sample = "../dummy_sample.csv"

    # Instantiate mgc class
    MGC = MusicGenreClassification(AudioFeatureExtraction, Classifier, music_dataset_path, setting_file)

    # Apply feature extraction and write out csv file if it does not exist
    if feature_extraction is True:
        # Apply feature extraction to all audio files
        print("Start feature extraction")
        extracted_feature_dataframe = MGC.feature_extraction()
        # Apply data process
        clean_dataframe = MGC.data_process(extracted_feature_dataframe)
        train_data, test_data, train_label, test_label = MGC.make_dataset(clean_dataframe, output_data_directory)
    else:
        # Read data from directory
        train_data, test_data, train_label, test_label = MGC.read_dataset(input_data_directory)

    if training is True:
        # Training model
        model = MGC.training(train_data, train_label, model_directory_path)
    else:
        # Load model
        model = MGC.CLF.load_model(model_file)

    # Test system
    accuracy = MGC.test(model, test_data, test_label)

    # Make prediction
    dummy_dataframe = FileUtil.csv2dataframe(dummy_sample)
    prediction_array = MGC.predict(model, dummy_dataframe)
    max_class = np.argmax(prediction_array)

    print(prediction_array)
    print(max_class)
    print("Start prediction")
    print("Final accuracy is {0}".format(accuracy))


if __name__ == "__main__":
    main()
