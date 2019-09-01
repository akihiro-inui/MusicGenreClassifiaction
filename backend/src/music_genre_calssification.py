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


class MusicGenreClassification:
    """
    # Content-based music genre classification
    # 1. Frame based feature extraction
    # 2. Data processing (Normalization, Encoding class label(string to number))
    # 3. Split into training data and test data (Shuffle order, train/test separation in "feature" directory)
    # 4. Train and save model (save trained model in "model" directory)
    # 5. Test classifier to get prediction accuracy
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

    def process_feature(self, directory_files_feature_dict: dict, label_list: list, normalize: bool):
        """

        :param directory_files_feature_dict:
        :param label_list:
        :param normalize:
        :return: expert_feature_2d_array: Extracted feature with expert system (Numpy 2D array)
        :return mel_spectrogram_array: Extracted mel-spectrogram (Numpy 3D array)
        :return list_array: Labels in numpy arrau
        """
        # Convert extracted feature and label to numpy array
        expert_feature_array, mel_spectrogram_array = self.AFE.dict2array(directory_files_feature_dict)
        label_array = np.array(label_list)

        # Normalize expert feature
        #if normalize is True:
        #    expert_feature_array = DataProcess.min_max_normalize(expert_feature_array)
        return expert_feature_array, mel_spectrogram_array, label_array

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
        :return Trained or pre-trained model
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
    pre_extracted_expert_feature_directory = "../feature/expert/genre"
    pre_extracted_2d_feature_directory = "../feature/mel_spectrogram/2019-09-01_02_58_54.985744"
    pre_trained_model_file = ""

    # Set Conditions
    use_expert_feature = False
    run_feature_extraction = False
    run_training = True

    # Instantiate mgc main class
    MGC = MusicGenreClassification(AudioDatasetMaker, AudioFeatureExtraction, Classifier,
                                   music_dataset_path="../../processed_data_alphanote",
                                   setting_file="../../config/master_config.ini")

    # Make label from genre names in processed_music_data
    MGC.make_label()

    # Feature extraction/ Load pre-extracted feature
    if run_feature_extraction is True:
        # Apply feature extraction
        directory_files_feature_dict, label_list = MGC.feature_extraction()

        # Apply data processing to extracted feature
        expert_feature_array, mel_spectrogram_array, list_array = MGC.process_feature(directory_files_feature_dict, label_list, MGC.cfg.normalize)

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
