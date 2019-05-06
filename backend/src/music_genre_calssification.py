#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Christmas 2018

@author: Akihiro Inui
"""

import os
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
    # 2. Data processing (Normalization, Encoding label(string to number))
    # 3. Make data set (Shuffle order, train/test separation, write train.csv/test.csv in "feature" directory with time)
    # 4. Train model / Save model (save trained model in "model" directory with time)
    # 5. Test classifier (test.csv)
    # 6. Make a prediction to a dummy data (dummy_data.csv)
    """

    def __init__(self, audio_dataset_maker: classmethod, audio_feature_extraction: classmethod, classifier: classmethod, \
                 dataset_path: str, setting_file: str):
        """
        Init
        :param  audio_dataset_maker:      audio dataset make class
        :param  audio_feature_extraction: audio feature extraction class
        :param  classifier:               classifier class
        :param  dataset_path:             path to data set
        :param  setting_file:             config file
        """
        self.ADM = audio_dataset_maker(setting_file)
        self.AFE = audio_feature_extraction(setting_file)
        self.CLF = classifier(setting_file)
        self.dataset_path = dataset_path
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
        Feature extraction to data set
        :return feature_2D_dataframe: extracted 2D feature in pandas data frame
        :return feature_3D_array:     extracted 3D feature in numpy array
        """
        # Extract all features from dataset and store them into dataframe, 3D array
        feature_2D_dataframe, feature_3D_array, label_list = self.AFE.extract_dataset(self.dataset_path, "mean")
        return feature_2D_dataframe, feature_3D_array, label_list

    def make_label(self, label_csv_file_path: str):
        """
        Make csv file where the classes label is written
        **** Make sure sort is True ****
        """
        # Get folder names
        label_list = FileUtil.get_folder_names(self.dataset_path, sort=True)
        assert len(label_list) == self.cfg.num_classes, "Number of the classes mismatch"
        FileUtil.list2csv(label_list, label_csv_file_path)

    def make_2D_dataset(self, feature_2D_dataframe, output_directory: str):
        """
        Make data set
        :param  feature_2D_dataframe:   extracted feature in data frame
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
        directory_name_with_time = os.path.join(output_directory, FileUtil.get_time())

        train_data, test_data, train_label, test_label = DataProcess.make_2D_dataset(feature_2D_dataframe, self.cfg.label_name,
                                                                                  self.cfg.test_rate, self.cfg.shuffle,
                                                                                  directory_name_with_time)
        return train_data, test_data, train_label, test_label

    def make_3D_dataset(self, feature_3D_array, label_list: list, output_directory: str):
        """
        Make data set
        :param  feature_3D_array:   extracted feature in data frame
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
        directory_name_with_time = os.path.join(output_directory, FileUtil.get_time())

        train_data, test_data, train_label, test_label = DataProcess.make_3D_dataset(feature_3D_array, label_list,
                                                                                  self.cfg.test_rate, self.cfg.shuffle,
                                                                                  directory_name_with_time)
        return train_data, test_data, train_label, test_label

    def read_2D_dataset(self, input_data_directory_with_date):
        """
        Read 2D feature data set
        :param  input_data_directory_with_date: name of the directory where train and test data exist
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Read data set
        train_data, test_data, train_label, test_label = DataProcess.read_2D_dataset(input_data_directory_with_date,
                                                                                  self.cfg.label_name)
        return train_data, test_data, train_label, test_label

    def read_3D_dataset(self, input_data_directory_with_date):
        """
        Read 3D data set
        :param  input_data_directory_with_date: name of the directory where train and test data exist
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Read data set
        train_data, test_data, train_label, test_label = DataProcess.read_3D_dataset(input_data_directory_with_date,
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
        processed_dataframe = DataProcess.factorize_label(processed_dataframe, self.cfg.label_name)
        return processed_dataframe

    def training(self, train_data, train_label, output_directory, visualize=None):
        """
        Train model and save it in the output_directory
        :param  train_data:  train data
        :param  train_label: train label
        :param  output_directory: output directory for model
        :param  visualize: True/False to visualize training history
        :return trained model
        """
        # Train classifier
        model = self.CLF.training(train_data, train_label, visualize)

        # Save mode with current time
        self.CLF.save_model(model, os.path.join(output_directory, FileUtil.get_time()))
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
    # File/folder path
    setting_file = "../../config/master_config.ini"
    music_dataset_path = "../../processed_music_data"
    model_directory_path = "../model"
    label_txt_filename = "label.csv"

    # To output extracted feature
    output_2D_feature_directory = "../feature/feature_2D"
    output_3D_feature_directory = "../feature/feature_3D"

    # Case of loading features
    input_2D_feature_directory = "../feature/feature_2D/2019-05-04_19:59:32.384607"
    input_3D_feature_directory = "../feature/feature_3D/2019-03-12_00:40:19.325268"

    # Case of loading pre-trained model
    model_file = "../model/2019-02-14_00:20:17.281506/mlp.h5"

    # Evaluate one file
    dummy_sample = "../dummy_data.csv"

    # Conditions
    feature_extraction = True
    training = True
    extract_2d_feature = True # If False, it extracts 3D feature

    # Instantiate mgc class
    MGC = MusicGenreClassification(AudioDatasetMaker, AudioFeatureExtraction, Classifier, music_dataset_path, setting_file)

    # Make label text file
    MGC.make_label(label_txt_filename)

    # Apply feature extraction and write out csv file if it does not exist
    if feature_extraction is True:
        # Apply feature extraction to all audio files
        print("Start feature extraction")
        feature_2D_dataframe, feature_3d_array, label_list = MGC.feature_extraction()

        if extract_2d_feature is True:
            # Apply data process to 2D array
            clean_dataframe = MGC.data_process(feature_2D_dataframe)
            train_data, test_data, train_label, test_label = MGC.make_2D_dataset(clean_dataframe, output_2D_feature_directory)
        else:
            # Apply data process to 3D array
            train_data, test_data, train_label, test_label = MGC.make_3D_dataset(feature_3d_array, label_list, output_3D_feature_directory)
    # Read data from directory
    else:
        if extract_2d_feature is True:
            train_data, test_data, train_label, test_label = MGC.read_2D_dataset(input_2D_feature_directory)
        else:
            train_data, test_data, train_label, test_label = MGC.read_3D_dataset(input_3D_feature_directory)

    # Training model
    if training is True:
        model = MGC.training(train_data, train_label, model_directory_path, visualize=True)
    # Load model
    else:
        model = MGC.CLF.load_model(model_file)

    # Test classifier
    accuracy = MGC.test(model, test_data, test_label)

    # Make prediction
    #dummy_dataframe = FileUtil.csv2dataframe(dummy_sample)
    prediction_array = MGC.predict(model, test_data)
    #max_class = np.argmax(prediction_array)
    predict_list = []
    for sample in list(prediction_array):
        predict_list.append(list(sample).index(max(list(sample))))



    #print(max_class)

    print("Start prediction")
    print("Final accuracy is {0}".format(accuracy))


if __name__ == "__main__":
    main()
