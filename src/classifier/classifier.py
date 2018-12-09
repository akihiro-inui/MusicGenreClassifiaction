#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""
# Import libraries/modules
from common.config_reader import ConfigReader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class Classifier:
    # Initialization
    def __init__(self, setting_file: str):
        """
        Initialization for parameters and classes
        :param setting_file: config file
        """
        # Load parameters from config file
        self.cfg = ConfigReader(setting_file)
        self.k = self.cfg.k

        # Initialize classifier selection
        self.classifier_selection_dict = self.cfg.section_reader("classifier_selection")
        self.classifier = self.__init_classifier_select()

    def __init_classifier_select(self) -> list:
        """
        Extract setting for classifier from config file
        :return name of selected classifier
        """
        for classifier, switch in self.classifier_selection_dict.items():
            if switch == "True":
                selected_classifier = classifier
        return selected_classifier

    def training(self, train_data, train_label):
        """
        Execute training
        :param  train_data: training data
        :param  train_label: test data
        :return model: trained model
        """
        assert self.classifier is not None, "No classifier selected"
        if self.classifier == "knn":
            model = Classifier.kNN(train_data, train_label, self.k)
        return model

    def predict(self, test_data, test_label, model) -> float:
        """
        Execute prediction
        :param  test_data: training data
        :param  test_label: test data
        :param  model: trained model
        :return prediction rate
        """
        return accuracy_score(test_label, model.predict(test_data))

    @staticmethod
    def kNN(train_data, train_label, k: int):
        """
        K Nearest Neighbors
        :param  train_data: training data
        :param  train_data: train data
        :param  k: number of k
        :return model: trained model
        """
        # Initialize our classifier
        model = KNeighborsClassifier(n_neighbors=k)
        # Train model
        model.fit(train_data, train_label)
        return model
