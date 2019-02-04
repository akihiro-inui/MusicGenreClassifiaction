#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:39:31 2017
Functions for image process
@author: Akihiro Inui
"""

import cv2
import numpy as np
from PIL import Image, ImageOps


class ImageUtil:

    @staticmethod
    def read_image_file(input_image_file_path: str):
        """
        Read image file return as numpy array
        :param  input_image_file_path: name of input image file
        :return Numpy array (height x width x number of colors)
        """
        return cv2.imread(input_image_file_path)

    @staticmethod
    def show_image_file(numpy_image_data):
        """
        Show image data in a new window
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        """
        cv2.imshow('image', numpy_image_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def write_image_file(numpy_image_data, output_image_file_path: str):
        """
        Write image file from numpy array
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        :param  output_image_file_path: name of input image file
        """
        cv2.imwrite(output_image_file_path, numpy_image_data)

    @staticmethod
    def resize(numpy_image_data, square_size: int):
        """
        Read image file return numpy array
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        :param  square_size: square image size after resize
        :return Numpy array (height x width x number of colors)
        """
        # Resize it into squared shape
        return cv2.resize(numpy_image_data, (square_size, square_size))

    @staticmethod
    def image2data(input_image_file_path: str):
        """
        Return the names of the folders in input_data_directory
        :param input_image_file_path:  path to the image data
        """
        # Convert image data format
        return Image.open(input_image_file_path).convert('RGB')

    @staticmethod
    def image2numpy(input_image_file_path: str):
        """
        Return the names of the folders in input_data_directory
        :param input_image_file_path:  path to the image data
        """
        # Convert image data into numpy data format
        return np.asarray(Image.open(input_image_file_path).convert('RGB'), np.uint8)

    @staticmethod
    def numpy2image(numpy_image_data):
        """
        Convert numpy image data into image format
        :type  numpy_image_data: numpy array
        :param numpy_image_data: image data with numpy array format
        """
        # Convert image data format from numpy array
        return Image.fromarray(np.uint8(numpy_image_data))

    @staticmethod
    def noise_image(numpy_image_data, mean=0, std=0.8):
        """
        Add Gaussian noise to the image file
        :type  numpy_image_data:  numpy array
        :param numpy_image_data:  image data in numpy array
        :type  mean:   int
        :param mean:   mean value for random noise
        :type  std:  int
        :param std:  parameter for Gaussian noise
        """
        # Add Gaussian noise to the image file
        noisy_image = numpy_image_data + np.random.normal(mean, std, numpy_image_data.shape)
        return np.clip(noisy_image, 0, 255)

    @staticmethod
    def blur_image(numpy_image_data, blur_size=10):
        """
        Add blur effect to the image file
        :type   numpy_image_data:  numpy array
        :param  numpy_image_data:  image data in numpy array
        :type   blur_size:   int
        :param  blur_size:   size of the blur effect
        :rtype  numpy array
        :return blurred image in numpy array
        """
        return cv2.blur(numpy_image_data, (blur_size, blur_size))

    @staticmethod
    def zoom_out_image(numpy_image_data, zoom_out_size=5):
        """
        Zoom out the image file
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        :type   zoom_out_size:   int
        :param  zoom_out_size:   zoom out proportion
        :rtype  numpy array
        :return zoomed out image in numpy array
        """
        # Reformat numpy array
        image_data = Image.fromarray(np.uint8(numpy_image_data))
        image_data.resize((int(image_data.width / zoom_out_size), int(image_data.height / zoom_out_size)))
        return np.asarray(Image.open(image_data).convert('RGB'), np.uint8)

    @staticmethod
    def zoom_in_image(numpy_image_data, zoom_in_size=5):
        """
        Zoom in the image file
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        :type   zoom_in_size:   int
        :param  zoom_in_size:   zoom in proportion
        :rtype  numpy array
        :return zoomed image in numpy array
        """
        image_data = Image.fromarray(np.uint8(numpy_image_data))
        image_data.resize((int(image_data.width * zoom_in_size), int(image_data.height * zoom_in_size)))
        return np.asarray(Image.open(image_data).convert('RGB'), np.uint8)

    @staticmethod
    def mirror_image(numpy_image_data):
        """
        Zoom in the image file
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        :rtype  numpy array
        :return : mirrored image in numpy array
        """
        return ImageOps.mirror(numpy_image_data)

    @staticmethod
    def flatten_image(numpy_image_data):
        """
        Square image, flatten it and normalize.
        :type   numpy_image_data: numpy array
        :param  numpy_image_data: image data in numpy array
        :rtype  numpy array
        :return flattened numpy array (vector)
        """
        # Flatten and normalise image data
        return numpy_image_data.flatten().astype(np.float32) / 255.0

