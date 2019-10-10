#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:01:21 2018

@author: Akihiro Inui
"""
import os
import csv
import pandas as pd
import numpy as np
import datetime
import ntpath


class FileUtil:

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
    def get_file_length(i_filename):
        """
        Count the number of lines in a file and return that count.
        :type i_filename string
        :param i_filename: the name of the file, which will have its lines counted.
        :rtype integer
        :return: the number of lines in the file.
        """
        i = 0
        with open(i_filename, "r", encoding='utf8') as input_file:
            for i, l in enumerate(input_file):
                pass
        return i + 1

    @staticmethod
    def replace_backslash(input_file_path: str):
        """
        Replace backslash in order to avoid Windows Mac convention mismatch
        """
        return input_file_path.replace('\\', '/')

    @staticmethod
    def rename_file(input_file_path: str):
        """
        Rename invalid file name
        """
        # Separate file path and name
        path, filename_base = ntpath.split(input_file_path)
        # Separate file name and extension
        filename, extension = os.path.splitext(filename_base)

        # Replace invalid char
        if filename.find("."):
            filename = filename.replace(".", "_")
        if filename.find(" "):
            filename = filename.replace(" ", "_")

        return os.path.join(path, filename+extension)

    @staticmethod
    def get_folder_names(directory_path: str, sort=True) -> list:
        """
        Return list of directories under the given path
        :param   directory_path: path to the directory
        :param   sort : True for alphabetical sort
        :return: list of folder names under the given path
        """
        assert FileUtil.is_invalid_directory(directory_path) is False, "Invalid directory path"

        folder_names_list = []
        for folder_name in os.listdir(directory_path):
            if not folder_name.startswith('.'):
                folder_names_list.append(folder_name)
        # Sort by alphabet
        if sort is True:
            folder_names_list = sorted(folder_names_list)
        return folder_names_list

    @staticmethod
    def get_file_names(directory_path: str, sort=True) -> list:
        """
        Return list of directories under the given path
        :param   directory_path: path to the directory
        :param   sort : True for alphabetical sort
        :return: list of file names under the given path
        """
        assert FileUtil.is_invalid_directory(directory_path) is False, "Invalid directory path"

        # Extract file/folder names under the input directory
        file_names_list = []
        for file_name in os.listdir(directory_path):
            if not file_name.startswith('.'):
                file_names_list.append(file_name)

        # Sort by alphabet
        if sort is True:
            file_names_list = sorted(file_names_list)
        return file_names_list

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
    def dataframe2csv(input_dataframe, output_filename: str):
        """
        # Write data frame to csv file
        :param  input_dataframe: input pandas data frame
        :param  output_filename: output csv file name
        """
        # Write dataframe as csv file
        input_dataframe.to_csv(output_filename, index=False)

    @staticmethod
    def list2csv(input_list: list, output_csv_file_path: str):
        """
        # Write list to csv file
        :param : input_list
        :param : output_text_file_path: csv file path to write out the input list
        """
        # Write list as csv file
        with open(output_csv_file_path, 'w') as f:
            for value in input_list:
                f.write(str(value) + ',')

    @staticmethod
    def list2text(input_list: list, output_text_file_path: str, headers=None):
        """
        # Write out list to text file
        :param : input_list
        :param : output_text_file_path: text file path to write out the input list
        :param : headers: header to be written in text file
        """
        # Make N number of comma as string
        if not headers:
            comma_str = ','
        else:
            comma_str = (len(headers)-1) * ','

        # Write out as text file
        with open(output_text_file_path, 'w') as f:
            # Write header on top
            if headers:
                for header in headers:
                    f.write("{}".format(header))
                    if header is not headers[-1]:
                        f.write(",".format(header))

                f.write("\n")
            # Write items in rows
            for item in input_list:
                f.write("{0}{1}\n".format(item, comma_str))

    @staticmethod
    def csv2dataframe(input_filename: str):
        """
        # Read csv file to data frame
        :param  input_dataframe: input csv file
        :return : output data frame
        """
        return pd.read_csv(input_filename)

    @staticmethod
    def get_time():
        """
        # Get current time and return as string
        :return : current time in string
        """
        return str(datetime.datetime.now()).replace(" ", "_")

    @staticmethod
    def save_3D_array(input_numpy_array, output_file_path: str):
        """
        # Save 3D numpy array
        :param  input_numpy_array: Input numpy array
        :param  output_file_path: Output file path
        """
        np.save(output_file_path, input_numpy_array)

    @staticmethod
    def load_3D_array(input_file_path):
        """
        # Load 3D numpy array
        :param  input_file_path: Input file path
        :return numpy array
        """
        return np.load(input_file_path)

    @staticmethod
    def text2list(input_text_file: str):
        """
        Read new line separated text file into list
        :param input_text_file: input text file
        :return: list of elements
        """
        if not FileUtil.is_valid_file(input_text_file):
            assert "Not valid file path"
        return [line.rstrip('\n') for line in open(input_text_file)]

    @staticmethod
    def commatext2list(input_text_file: str):
        """
        Read new line separated text file into list
        :param input_text_file: input text file
        :return: list of elements
        """
        if not FileUtil.is_valid_file(input_text_file):
            assert "Not valid file path"
        with open(input_text_file, 'r') as f:
            reader = csv.reader(f)
            return list(reader)
