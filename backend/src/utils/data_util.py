#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File reader/writer
@author: Akihiro Inui
ainui@jabra.com
"""

# Import libraries
import pandas as pd
import requests
import datetime
import os


class FileUtil:
    """
    # Functions for data manipulation
    """
    @staticmethod
    def is_valid_file(input_filename: str) -> bool:
        """
        # Confirm the filename is a valid file
        :param   input_filename: the name of the file to read in.
        :return  True if it is valid file and False if it is invalid file
        """
        return not FileUtil.is_invalid_file(input_filename)

    @staticmethod
    def is_invalid_file(input_filename: str) -> bool:
        """
        # Confirm the filename is a valid file
        :param   input_filename: the name of the file to read in.
        :return  True if it is invalid file and False if it is valid file
        """
        return not os.path.isfile(input_filename)

    @staticmethod
    def is_valid_directory(directory_path: str) -> bool:
        """
        # Check if the directory path is a valid or invalid directory
        :param directory_path:
        :return: True if it is valid directory and False if it is invalid directory
        """
        return not FileUtil.is_invalid_directory(directory_path)

    @staticmethod
    def is_invalid_directory(directory_path: str) -> bool:
        """
        # Check if the directory path is a valid or invalid directory
        :param  directory_path:
        :return: True if it is invalid directory and False if it is valid directory
        """
        return not os.path.isdir(directory_path)

    @staticmethod
    def excel2dataframe(input_file_path: str):
        """
        # Read excel file into pandas dataframe
        :param  input_file_path: input excel file
        :return pandas data frame
        """
        # Read excel and write out as csv file
        return pd.read_excel(input_file_path, index=False)

    @staticmethod
    def get_unique(input_dataframe, column_name: str):
        """
        # Get unique values in one column
        :param  input_dataframe: input pandas data frame
        :param  column_name: column name
        """
        unique_value = input_dataframe[column_name].unique()
        return unique_value

    @staticmethod
    def replace_comma(input_dataframe):
        """
        # Replace all commas in the data frame
        :param  input_dataframe: input pandas data frame
        :return input_dataframe: output pandas data frame without comma
        """

        # Replace comma for each column
        for column in input_dataframe.columns:
            input_dataframe[column] = input_dataframe[column].apply(
                lambda x: str(x.replace(",", " ")) if type(x) is str else x)
        return input_dataframe

    @staticmethod
    def replace_newline(input_dataframe):
        """
        # Replace all new lines in the data frame
        :param  input_dataframe: input pandas data frame
        :return input_dataframe: output pandas data frame without new line
        """
        # Replace comma for each column
        for column in input_dataframe.columns:
            input_dataframe[column] = input_dataframe[column].apply(
                lambda x: str(x.replace("\n", " ")) if type(x) is str else x)
            input_dataframe[column] = input_dataframe[column].apply(
                lambda x: str(x.replace("\r", " ")) if type(x) is str else x)
        return input_dataframe

    @staticmethod
    def write2csv(input_dataframe, output_filename: str):
        """
        # Write data frame to csv file
        :param  input_dataframe: input pandas data frame
        :param  output_filename: output csv file name
        """
        input_dataframe.to_csv(output_filename, index=False)

    @staticmethod
    def get_time():
        """
        # Get current time and return as string
        :return : current time in string
        """
        time = str(datetime.datetime.now()).replace(":", "_")
        time = time.replace(".", "_")
        return time.replace(" ", "_")

    @staticmethod
    def is_downloadable(url):
        """
        Does the url contain a downloadable resource
        """
        h = requests.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            return False
        if 'html' in content_type.lower():
            return False
        return True

    @staticmethod
    def download_data(url):
        """
        # Download file from the given url
        """
        # Check if the file is downloadable
        h = requests.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            return False
        if 'html' in content_type.lower():
            return False
        file_name = os.path.basename(url)
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(file_name, 'wb') as file:
                for chunk in res.iter_content(chunk_size=1024):
                    file.write(chunk)

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
            if type(feature) is list or tuple:
                for element in feature:
                    flat_list.append(element)
            else:
                flat_list.append(feature)
        return flat_list

