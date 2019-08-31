#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Christmas 2018
@author: Akihiro Inui
"""

import os
import numpy as np
from backend.src.utils.file_utils import FileUtil
from backend.src.classifier.classifier_wrapper import Classifier
from backend.src.common.config_reader import ConfigReader
from backend.src.data_process.data_process import DataProcess
from backend.src.feature_extraction.audio_feature_extraction import AudioFeatureExtraction
from backend.src.data_process.audio_dataset_maker import AudioDatasetMaker
import matplotlib.pyplot as plt


class MusicGenreClassification:
    """
    # Content-based music genre classification
    # 1. Frame based feature extraction
    # 2. Data processing (Normalization, Encoding class label(string to number))
    # 3. Split into training data and test data (Shuffle order, train/test separation, write train.csv/test.csv in "feature" directory with time)
    # 4. Train and save model (save trained model in "model" directory with time)
    # 5. Test classifier to get prediction accuracy (to test.csv)
    """

    def __init__(self, audio_dataset_maker: classmethod, audio_feature_extraction: classmethod, classifier: classmethod,
                 music_dataset_path: str, setting_file: str):
        """
        Init
        :param  audio_dataset_maker:      audio dataset make class
        :param  audio_feature_extraction: audio feature extraction class
        :param  classifier:               classifier class
        :param  music_dataset_path:             path to data set
        :param  setting_file:             config file
        """
        self.ADM = audio_dataset_maker(setting_file)
        self.AFE = audio_feature_extraction(setting_file)
        self.CLF = classifier(setting_file)
        self.dataset_path = music_dataset_path
        self.cfg = ConfigReader(setting_file)
        self.setting_file = setting_file

    def make_audio_dataset(self, input_dataset_path: str, output_dataset_path: str):
        """
        Process dataset in order to keep the consistency in audio length, sampling rate and level
        :param  input_dataset_path: original dataset path
        :param  output_dataset_path: new dataset path
        """
        self.ADM.process_dataset(input_dataset_path, output_dataset_path)

    def feature_extraction(self):
        """
        Feature extraction to data set. Extract expert system feature and Mel-spectrogram
        :return directory_files_feature_dict: dictionary of extracted features from all audio files in dataset folder
        {key: name of directory, value: list of file names {key: file name, value: list of extracted features}}
        :return directory_files_mel_spectrogram_dict: dictionary of Mel-spectrogram from all audio files in dataset folder
        {key: name of directory, value: list of file names {key: file name, value: Mel-spectrogram}}
        :return label_list: list of numerical label vector
        """
        # Extract all features from dataset and store them into dictionary
        return self.AFE.extract_dataset(self.dataset_path)

    def make_label(self):
        """
        Make csv file where the classes label is written
        **** Make sure sort is True ****
        """
        # Get folder names
        label_list = FileUtil.get_folder_names(self.dataset_path, sort=True)
        assert len(label_list) == self.cfg.num_classes, "Number of the classes mismatch"
        FileUtil.list2csv(label_list, "../label.csv")

    def process_feature(self, directory_files_feature_dict, label_list):
        # Convert extracted feature and label to numpy array
        expert_feature_2d_array, mel_spectrogram_3d_array = self.AFE.dict2array(directory_files_feature_dict)
        list_array = np.array(label_list)
        return expert_feature_2d_array, mel_spectrogram_3d_array, list_array

    def make_dataset(self, feature_array, label_array, output_directory: str):
        """
        Make data set
        :param  feature_array: extracted feature in 2D numpy array or 3D numpy array
        :param  label_array: labels in numpy array
        :param  output_directory: output directory to write out the train and test data
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Make directory if it does not exist
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        # Get time and make a new directory name
        directory_name_with_time = os.path.join(output_directory, FileUtil.get_time().replace(":", "_"))

        train_data, test_data, train_label, test_label = DataProcess.make_dataset_from_array(feature_array, label_array,
                                                                                  self.cfg.test_rate, self.cfg.shuffle,
                                                                                  directory_name_with_time)
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

        # Centerize data and apply standardization to data frame
        # processed_dataframe = DataProcess.normalize_dataframe(processed_dataframe, self.cfg.label_name)
        centerized_dataframe, mean_list = DataProcess.centerize_dataframe(processed_dataframe, self.cfg.label_name)
        cleaned_dataframe, std_list = DataProcess.standardize_dataframe(centerized_dataframe, self.cfg.label_name)

        # Factorize label
        cleaned_dataframe = DataProcess.factorize_label(cleaned_dataframe, self.cfg.label_name)

        # Write out mean values
        FileUtil.list2csv(mean_list, "../mean_list.csv")
        FileUtil.list2csv(std_list, "../std_list.csv")
        return cleaned_dataframe

    def training(self, run_training: bool, train_data, train_label,
                 model_file=None, output_model_directory_path=None, visualize=True):
        """
        Top level wrapper for training model
        :param run_training: True to run training model. If False, it loads pre-trained model
        :param train_data: Training data
        :param train_label: Training label
        :param model_file: Pre-trained model file path to load
        :param output_model_directory_path: Output directory to save trained model
        :param visualize: Set True to visualize training history
        :return Trained/pre-trained model
        """
        # Training model
        if run_training is True:
            model = self.CLF.training(train_data, train_label, visualize)
            self.CLF.save_model(model, os.path.join(output_model_directory_path, FileUtil.get_time().replace(":", "_")))
        # Load model
        else:
            model = self.CLF.load_model(model_file)
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
    # Case of loading pre-extracted features and/or pre-trained feature
    pre_extracted_expert_feature_directory = "../feature/expert/2019-08-27_04_07_56.098135"
    pre_extracted_2d_feature_directory = "../feature/mel_spectrogram/2019-08-30_07_30_14.149285"
    pre_trained_model_file = "../model/2019-08-06_06_37_17.286254/kNN.pickle"

    # Conditions
    use_expert_feature = False
    run_feature_extraction = True
    run_training = True

    # Instantiate mgc main class
    MGC = MusicGenreClassification(AudioDatasetMaker, AudioFeatureExtraction, Classifier,
                                   music_dataset_path="../../original_processed_music_data",
                                   setting_file="../../config/master_config.ini")

    # Make label from genre names in processed_music_data
    MGC.make_label()

    # Feature extraction/ Load pre-extracted feature
    if run_feature_extraction is True:
        directory_files_feature_dict, label_list = MGC.feature_extraction()
        # Data processing to extracted feature
        expert_feature_array, mel_spectrogram_array, list_array = MGC.process_feature(directory_files_feature_dict, label_list)

        # Run feature extraction or load pre-extracted feature
        if use_expert_feature is True:
            train_data, test_data, train_label, test_label = MGC.make_dataset(expert_feature_array, list_array, "../feature/expert")
        else:
            train_data, test_data, train_label, test_label = MGC.make_dataset(mel_spectrogram_array, list_array, "../feature/mel_spectrogram")
    else:
        # Load pre-extracted feature
        if use_expert_feature is True:
            train_data, test_data, train_label, test_label = DataProcess.read_dataset_from_array(pre_extracted_expert_feature_directory)
        else:
            train_data, test_data, train_label, test_label = DataProcess.read_dataset_from_array(pre_extracted_2d_feature_directory)

    # Run training or load pre-trained model
    model = MGC.training(run_training, train_data, train_label, pre_trained_model_file, output_model_directory_path="../model")

    # Test model performance
    accuracy = MGC.test(model, test_data, test_label)
    print("Test accuracy is {0}% \n".format(accuracy))


if __name__ == "__main__":
    main()
