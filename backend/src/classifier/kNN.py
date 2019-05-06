#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: Akihiro Inui
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import os


class kNN:
    """
    K Nearest Neighbors
    """
    def __init__(self, k: int):
        """
        Init
        :param  k: number of k
        """
        self.k = k

    def training(self, train_data, train_label, visualize=None):
        """
        Train model
        :param  train_data: training data
        :param  train_label: training label
        :return model: trained model
        """
        # Initialize our classifier
        model = KNeighborsClassifier(n_neighbors=self.k)
        # Train model
        model.fit(train_data, train_label)
        return model

    def save_model(self, model, output_directory: str):
        """
        Save model
        :param  model: trained model
        :param  model_file_name: name of model file to load
        """
        # Pickle model
        with open(os.path.join(output_directory, "kNN.pickle"), mode='wb') as fp:
            pickle.dump(model, fp, protocol=2)

    def test(self, model, test_data, test_label):
        """
        Make test with trained model
        :param  model: trained model
        :param  test_data: test data
        :param  test_label: test label
        :return : test score
        """
        # Make a prediction
        return accuracy_score(test_label, model.predict(test_data))

    def predict(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data
        :return prediction array with probability
        """
        return model.predict_proba(target_data)

