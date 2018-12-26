#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: Akihiro Inui
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class kNN:
    """
    K Nearest Neighbors
    :param  k: number of k
    """

    def __init__(self, k: int):
        """
        Init
        :param  k: number of k
        """
        self.k = k

    def training(self, train_data, train_label):
        """
        Train model
        :param  train_data: training data
        :param  train_label: train label
        :return model: trained model
        """
        # Initialize our classifier
        model = KNeighborsClassifier(n_neighbors=self.k)
        # Train model
        model.fit(train_data, train_label)
        return model

    def predict(self, model, test_data, test_label):
        """
        Make prediction with trained model
        :param  model: trained model
        :param  test_data: test data
        :param  test_label: test label
        :return accuracy: prediction score
        """
        # Make a prediction
        return accuracy_score(test_label, model.predict(test_data))
