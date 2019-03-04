#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 2018

@author: Akihiro Inui
"""

import numpy as np


class DataCleaner:

    @staticmethod
    def clean_data(input_dataframe):
        """
        Replace Inf to NaN, then replace NaN to 0
        :param  input_dataframe: input pandas dataframe
        :return clean dataframe
        """
        # Copy input dataframe
        clean_dataframe = input_dataframe.copy()

        # Replace Inf to NaN
        clean_dataframe=clean_dataframe.replace([np.inf, -np.inf], np.nan)
        clean_dataframe=clean_dataframe.replace([np.inf, -np.inf], np.nan)

        # Replace NaN to 0
        clean_dataframe = clean_dataframe.fillna(0)
        return clean_dataframe
