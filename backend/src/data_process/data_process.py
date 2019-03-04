#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 2018

@author: Akihiro Inui
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from backend.src.utils.file_utils import FileUtil
from backend.src.data_process.datacleaner import DataCleaner


class DataProcess:

    @staticmethod
    def segment_feature_dict(input_feature_dict: dict) -> dict:
        """
        Segment features which are stored in dictionary
        e.g. {key:mfcc, value: list of numbers} -> {key:mfcc1, value: value1, key:mfcc2, value: value2....}
        :param input_feature_dict: {key: feature name, value: array/list or single value}
        :return segmented_feature_dict: {key: feature name, value: single value}
        """
        # Initialize new dictionary
        segmented_feature_dict = {}
        for feature_name, feature_vector in input_feature_dict.items():
            if np.size(feature_vector) > 1:
                counter = 1
                # Store feature for each element
                for value in feature_vector:
                    new_feature_name = "{0}_{1}".format(feature_name, counter)
                    segmented_feature_dict[new_feature_name] = value
                    counter += 1
            else:
                # Store feature as it is if one element
                segmented_feature_dict[feature_name] = feature_vector
        return segmented_feature_dict

    @staticmethod
    def clean_feature_dict(input_file_feature_dict: dict) -> dict:
        """
        Segment features which are stored in dictionary for all files
        :param: input_file_feature_dict: {key: file name, value: dictionary{key: feature name, value: single value}}
        """
        new_file_feature_dict = {}
        # Segment feature values for all files
        for file_name, feature_dict in input_file_feature_dict.items():
            new_file_feature_dict[file_name] = DataProcess.segment_feature_dict(feature_dict)
        return new_file_feature_dict

    @staticmethod
    def dict2dataframe(input_dict: dict, segment_feature: False):
        """
        Convert dictionary to data frame
        :param: input_dict: {key: file name, value: dictionary{key: feature name, value: single value}}
        :param: segment_feature: If True, segment all feature vector elements (Case of numerical feature)
        """
        if segment_feature is True:
            new_feature_dict = DataProcess.clean_feature_dict(input_dict)
            feature_dataframe = pd.DataFrame.from_dict(new_feature_dict, orient='index')
        else:
            feature_dataframe = pd.DataFrame.from_dict(input_dict, orient='index')
        return feature_dataframe

    @staticmethod
    def add_label(input_dataframe, label_name: str):
        """
        Add label to the end of input data frame
        :param: input_dataframe: input pandas dataframe
        :param: label_name: Name of label to add
        """
        return input_dataframe.assign(category=label_name)

    @staticmethod
    def drop_column(input_dataframe, column: int or str):
        """
        Remove column from data frame
        :param: input_dataframe: input pandas dataframe
        :param: column: Name or number of column to be removed
        """
        if type(column) is int:
            new_dataframe = input_dataframe.drop(input_dataframe.columns[[column]], axis=1)
        else:
            new_dataframe = input_dataframe.drop(columns=column)
        return new_dataframe

    @staticmethod
    def get_unique(input_dataframe, column_name: str):
        """
        # Get unique values in one column
        :param  input_dataframe: input pandas data frame
        :param  column_name: column name
        :return unique attribute of the column in dataframe
        """
        return input_dataframe[column_name].unique()

    @staticmethod
    def folder2label(input_data_directory: str) -> list:
        """
        # Get unique values in one column
        :param  input_data_directory: input data directory where data folder with the label name exist
        :return list of folder names
        """
        return FileUtil.get_file_names(input_data_directory)

    @staticmethod
    def factorize_label(input_dataframe, column_name: str):
        """
        # Factorize str label to num label
        :param  input_dataframe : input pandas data frame
        :param  column_name : column name to factorize
        :return factorized_dataframe : data frame with factorized label
        :return label_list : list which stores label names
        """
        # Make a copy of the data frame
        factorized_dataframe = input_dataframe.copy()

        # Factorize string label
        factorized_column, unique_names = pd.factorize(factorized_dataframe[column_name])
        factorized_dataframe[column_name] = factorized_column

        return factorized_dataframe

    @staticmethod
    def normalize_dataframe(input_dataframe, label_name: str or list):
        """
        # Apply normalization to data frame
        :param  input_dataframe: input pandas data frame
        :param  label_name: exception column
        :return normalized_dataframe: normalized data frame
        """
        # Make a copy of the data frame
        normalized_dataframe = input_dataframe.copy()
        # Extract only data part
        data = normalized_dataframe[normalized_dataframe.columns[normalized_dataframe.columns != label_name]]

        # Replace NaN/Inf as 0
        clean_data = DataCleaner.clean_data(data)

        # Apply normalizaion to all columns
        for feature_name in data.columns:
            max_value = clean_data[feature_name].max()
            min_value = clean_data[feature_name].min()
            normalized_dataframe[feature_name] = (clean_data[feature_name] - min_value) / (max_value - min_value)

        # Put back label column
        normalized_dataframe[label_name] = input_dataframe[label_name]

        return normalized_dataframe

    @staticmethod
    def standardize_dataframe(input_dataframe, label_name: str or list):
        """
        # Apply standardization
        :param  input_dataframe: input pandas data frame
        :param  label_name: exception column
        :return standardized_dataframe: standardized data frame
        """
        # Make a copy of the data frame
        standardized_dataframe = input_dataframe.copy()
        # Extract only data part
        data = standardized_dataframe[standardized_dataframe.columns[standardized_dataframe.columns != label_name]]

        # Clean data
        clean_data = DataCleaner.clean_data(data)
        # Apply standardization to all columns
        for feature_name in clean_data.columns:
            standardized_dataframe[feature_name] = (clean_data[feature_name] - np.mean(clean_data[feature_name])) / np.std(clean_data[feature_name])

        # Put back label column
        standardized_dataframe[label_name] = input_dataframe[label_name]

        return standardized_dataframe

    @staticmethod
    def data_label_split(input_dataframe, label_name: str):
        """
        # Split data frame into data and label
        :param  input_dataframe: input pandas data frame
        :param  label_name: name of label column
        :return
        """
        # Make a copy of the dataframe
        output_dataframe = input_dataframe.copy()
        # Split data and label and store the information in the data frame
        label = output_dataframe[label_name]
        data = output_dataframe[output_dataframe.columns[output_dataframe.columns != label_name]]
        return label, data

    @staticmethod
    def train_test_split(input_dataframe, label_name: str, test_size: float, shuffle: False):
        """
        # Split data frame into data and label
        :param  input_dataframe: input pandas data frame
        :param  label_name: name of label column
        :param  test_size: name of label column
        :param  shuffle: Bool: True for shuffle
        :return train_data:  train data
        :return train_label: train label
        :return test_data:   test data
        :return test_label:  test label
        """
        # Case of data frame has data and target information
        if input_dataframe.target and input_dataframe.data:
            # Train/Test separation
            train_data, test_data, train_label, test_label = train_test_split(input_dataframe.data,
                                                                              input_dataframe.target,
                                                                              test_size=test_size,
                                                                              shuffle=shuffle)
        else:
            # Train/Test separation
            train_data, test_data, train_label, test_label = train_test_split(
                input_dataframe[input_dataframe.columns[input_dataframe.columns != label_name]],
                input_dataframe[label_name],
                test_size=test_size,
                shuffle=shuffle)
        return train_data, test_data, train_label, test_label

    @staticmethod
    def make_dataset(dataframe, label_name: str, test_size: float, shuffle: bool = False, output_directory: str = None):
        """
        Make data set and save the train/test data under output_directory
        1. Split label and data from the input dataframe
        2. Split them into train/test data and train/test label
        3. If output_directory is given, writes out them in csv format to the directory with current time
        4. Data and label are respectively saved as "train.csv" and "test.csv"

        :param  dataframe:   extracted feature in data frame
        :param  label_name:  name of label column in data frame
        :param  test_size:   size of test data set
        :param  shuffle:     set True for randomisation
        :param  output_directory: output directory to save train and test data
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
                                                                          shuffle=shuffle)

        # Write out test and train data as csv files if output directory name is given
        if output_directory:
            # Return error of the target directory already exist
            assert os.path.exists(output_directory) is False, "Target output data folder already exist"
            os.mkdir(output_directory)
            FileUtil.dataframe2csv(pd.concat([train_data, train_label], axis=1),
                                   os.path.join(output_directory, "train.csv"))
            FileUtil.dataframe2csv(pd.concat([test_data, test_label], axis=1),
                                   os.path.join(output_directory, "test.csv"))
        else:
            print("Data set is created but not saved")

        return train_data, test_data, train_label, test_label

    @staticmethod
    def read_dataset(input_data_directory_with_date: str, label_name: str):
        """
        Read data set under the given directory, return data and label
        :param  input_data_directory_with_date: input data directory with time where train.csv and test.csv exist
        :param  label_name: name of label column in dataframe
        :return data: data in dataframe
        :return label: label in series
        """
        # Get file names
        train_file_path = os.path.join(input_data_directory_with_date, "train.csv")
        test_file_path = os.path.join(input_data_directory_with_date, "test.csv")

        # Check if the given data set exist
        FileUtil.is_valid_file(train_file_path)
        FileUtil.is_valid_file(test_file_path)

        # Read csv file
        train_dataframe = FileUtil.csv2dataframe(train_file_path)
        test_dataframe = FileUtil.csv2dataframe(test_file_path)

        # Split data and label
        train_label, train_data = DataProcess.data_label_split(train_dataframe, label_name)
        test_label, test_data = DataProcess.data_label_split(test_dataframe, label_name)

        return train_data, test_data, train_label, test_label

    @staticmethod
    def flatten_list(input_list: list) -> list:
        """
        # Flatten list elements in the input list
        :param  input_list: Input list to be flattened
        :return flat_list: Flattened list
        """
        # Prepare empty list to store elements
        flat_list = []

        # Add element one by one
        for feature in input_list:
            if type(feature) is list:
                for element in feature:
                    flat_list.append(element)
            else:
                flat_list.append(feature)
        return flat_list
