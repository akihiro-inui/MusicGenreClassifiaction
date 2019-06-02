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
        Feature extraction to data set
        :return feature_2D_dataframe: extracted 2D feature in pandas data frame
        :return feature_3D_array:     extracted 3D feature in numpy array
        """
        # Extract all features from dataset and store them into dataframe, 3D array
        feature_2D_dataframe, feature_3D_array, label_list = self.AFE.extract_dataset(self.dataset_path, "mean")
        return feature_2D_dataframe, feature_3D_array, label_list

    def make_label(self):
        """
        Make csv file where the classes label is written
        **** Make sure sort is True ****
        """
        # Get folder names
        label_list = FileUtil.get_folder_names(self.dataset_path, sort=True)
        assert len(label_list) == self.cfg.num_classes, "Number of the classes mismatch"
        FileUtil.list2csv(label_list, "../label.csv")

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
        directory_name_with_time = os.path.join(output_directory, FileUtil.get_time().replace(":", "_"))

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
        directory_name_with_time = os.path.join(output_directory, FileUtil.get_time().replace(":", "_"))

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

        # Centerize data and apply standardization to data frame
        #processed_dataframe = DataProcess.normalize_dataframe(processed_dataframe, self.cfg.label_name)
        centerized_dataframe, mean_list = DataProcess.centerize_dataframe(processed_dataframe, self.cfg.label_name)
        cleaned_dataframe, std_list = DataProcess.standardize_dataframe(centerized_dataframe, self.cfg.label_name)

        # Factorize label
        cleaned_dataframe = DataProcess.factorize_label(cleaned_dataframe, self.cfg.label_name)

        # Write out mean values
        FileUtil.list2csv(mean_list, "../mean_list.csv")
        FileUtil.list2csv(std_list, "../std_list.csv")
        return cleaned_dataframe

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
        self.CLF.save_model(model, os.path.join(output_directory, FileUtil.get_time().replace(":", "_")))
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

    def feature_extraction_selector(self, run_feature_extraction: bool, use_2d_feature: bool,
                                    pre_extracted_2d_feature_directory=None, pre_extracted_3d_feature_directory=None,
                                    output_2d_feature_directory=None, output_3d_feature_directory=None):
        """
        Top level wrapper for feature extraction
        :param run_feature_extraction: True to extract feature. If False, it loads pre-extracted feature from directory
        :param use_2d_feature: If set to True, it extract 2D feature and to False, it extracts 3D feature
        :param pre_extracted_2d_feature_directory: Path to directory where pre-extracted 2D feature exist
        :param pre_extracted_3d_feature_directory: Path to directory where pre-extracted 3D feature exist
        :param output_2d_feature_directory: Output directory for 2D feature extraction
        :param output_3d_feature_directory: Output directory for 3D feature extraction
        """
        # Apply feature extraction
        if run_feature_extraction is True:

            # Apply feature extraction to all audio files
            print("Start feature extraction")
            feature_2d_dataframe, feature_3d_array, label_list = self.feature_extraction()

            # Apply data process to extracted feature
            if use_2d_feature is True:
                # Apply data process to 2D array
                clean_dataframe = self.data_process(feature_2d_dataframe)
                train_data, test_data, train_label, test_label = self.make_2D_dataset(clean_dataframe, output_2d_feature_directory)
            else:
                # Apply data process to 3D array
                train_data, test_data, train_label, test_label = self.make_3D_dataset(feature_3d_array, label_list, output_3d_feature_directory)

        # Load pre-extracted feature from directory
        else:
            if use_2d_feature is True:
                # Load 2D feature
                train_data, test_data, train_label, test_label = self.read_2D_dataset(pre_extracted_2d_feature_directory)
            else:
                # Load 3D feature
                train_data, test_data, train_label, test_label = self.read_3D_dataset(pre_extracted_3d_feature_directory)

        return train_data, test_data, train_label, test_label

    def training_selector(self, run_training: bool, train_data, train_label,
                          model_file=None, output_model_directory_path=None):
        """
        Top level wrapper for training model
        :param run_training: True to run training model. If False, it loads pre-trained model
        :param train_data: Training data
        :param train_label: Training label
        :param model_file: Pre-trained model file path to load
        :param output_model_directory_path: Output directory to save trained model
        """
        # Training model
        if run_training is True:
            model = self.training(train_data, train_label, output_model_directory_path, visualize=True)

        # Load model
        else:
            model = self.CLF.load_model(model_file)
        return model


def main():

    # Case of loading pre-extracted features / pre-trained feature
    pre_extracted_2d_feature_directory = '../feature/feature_2D/2019-05-17_18_36_31.283054'
    pre_extracted_3d_feature_directory = "../feature/feature_3D/2019-05-07_22_06_38.549108"
    pre_trained_model_file = "../model/2019-02-14_00:20:17.281506/mlp.h5"

    # Evaluate one file
    dummy_sample = "../dummy_data.csv"

    # Conditions
    run_feature_extraction = True
    run_training = True
    use_2d_feature = True

    # Instantiate mgc main class
    MGC = MusicGenreClassification(AudioDatasetMaker, AudioFeatureExtraction, Classifier,
                                   music_dataset_path="../../processed_music_data",
                                   setting_file="../../config/master_config.ini")

    # Make label from genre names in processed_music_data
    MGC.make_label()

    # Run feature extraction or load pre-extracted feature
    train_data, test_data, train_label, test_label = MGC.feature_extraction_selector(run_feature_extraction, use_2d_feature,
                                                                                     pre_extracted_2d_feature_directory,
                                                                                     pre_extracted_3d_feature_directory,
                                                                                     output_2d_feature_directory="../feature/feature_2D",
                                                                                     output_3d_feature_directory="../feature/feature_3D")

    # Run training or load pre-trained model
    model = MGC.training_selector(run_training, train_data, train_label, pre_trained_model_file, output_model_directory_path="../model")

    # Test model performance
    print("Start Testing \n")
    accuracy = MGC.test(model, test_data, test_label)
    print("Final accuracy is {0}% \n".format(accuracy*100))

    # Make prediction
    print("Start prediction \n")
    #dummy_dataframe = FileUtil.csv2dataframe(dummy_sample)
    prediction_array = MGC.predict(model, test_data)
    #max_class = np.argmax(prediction_array)
    predict_list = []
    for sample in list(prediction_array):
        predict_list.append(list(sample).index(max(list(sample))))
    print(predict_list)


if __name__ == "__main__":
    main()
