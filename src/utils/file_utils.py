"""
Write file related utilities like checking valid file, valid directory, zip files etc.
Do not use for it writing and reading files where line counting and other variables are used at class level
use file_reader_writer.py for that purpose

@author: Dinakar Maurya
"""
import os


class FileUtils:

    @staticmethod
    def is_valid_file(i_filename):
        """
        Confirm the filename is a valid file
        :type i_filename: str
        :param i_filename: the name of the file to read in.
        :rtype: Boolean
        :return: True if it is valid file and False if it is invalid file
        """
        return not FileUtils.is_invalid_file(i_filename)

    @staticmethod
    def is_invalid_file(i_filename):
        """
        Confirm the filename is a valid file
        :type i_filename: str
        :param i_filename: the name of the file to read in.
        :rtype: Boolean
        :return: True if it is invalid file and False if it is valid file
        """
        return not os.path.isfile(i_filename)

    @staticmethod
    def is_valid_directory(directory_path):
        """
        check if the directory path is a valid or invalid directory
        :type directory_path: str
        :param directory_path:
        :rtype Boolean
        :return: True if it is valid directory and False if it is invalid directory
        """
        return not FileUtils.is_invalid_directory(directory_path)

    @staticmethod
    def is_invalid_directory(directory_path):
        """
        check if the directory path is a valid or invalid directory
        :type directory_path: str
        :param directory_path:
        :rtype Boolean
        :return: True if it is invalid directory and False if it is valid directory
        """
        return not os.path.isdir(directory_path)

    @staticmethod
    def write_text_on_file(output_directory, filename, text):
        """
        write the text data on given file name, text data can be string, json etc.
        :type output_directory:str
        :param output_directory: directory where the file needs to be written
        :type filename: str
        :param filename: name of the file to write the text
        :type text:str
        :param text: text to write on file
        :return: None
        """
        if FileUtils.is_invalid_directory(output_directory):
            return False

        with open(os.path.join(output_directory, filename), 'w') as user_output_file:
            user_output_file.write(str(text))

    @staticmethod
    def file_length(i_filename):
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