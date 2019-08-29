#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3, June
Dataset maker using torchvision
@author: Akihiro Inui
"""
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class Torch2DDataset(Dataset):

    def __init__(self, dataset_root_directory: str, input_csv_file: str, transform=None):
        """
        Init
        :param  dataset_root_directory: validation rate
        :param  image_size:  square shape image size
        :param  input_csv_file: number of colors
        """
        # Read dataset information from csv file (filename, label)
        self.dataset_root_directory = dataset_root_directory
        self.dataframe = pd.read_csv(input_csv_file)
        self.transform = transform

    def __len__(self):
        """
        Length of data
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        """
        Extract one data
        :param  idx: Data index
        :return img_data: Torch tensor image data
        :return label: Label in digit
        """
        # Read image file path and label from dataframe
        label = self.dataframe.loc[idx, "label"]
        filename = self.dataframe.loc[idx, "filename"]
        filename = os.path.join(self.dataset_root_directory, filename)

        # Read image file
        img_data = Image.open(filename)

        # Apply pre-process to image file
        if self.transform:
            img_data = self.transform(img_data)
        return img_data, label
