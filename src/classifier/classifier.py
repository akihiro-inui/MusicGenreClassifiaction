#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

# Import libraries/modules
from classifier.kNN import kNN
from classifier.mlp import MLP
from common.config_reader import ConfigReader
import matplotlib.pyplot as plt
import os


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
        self.validation_rate = self.cfg.validation_rate

        # Initialize classifier selection
        self.classifier_selection_dict = self.cfg.section_reader("classifier_selection")
        self.selected_classifier = self.__init_classifier_select()
        self.classifier = self.__init_classifier()

    def __init_classifier_select(self) -> str:
        """
        Load config file and select classifier
        :return selected classifier name
        """
        # Search which classifier is True
        self.selected_classifier = [classifier for classifier, switch in self.classifier_selection_dict.items() if
                                    switch == "True"]
        assert len(self.selected_classifier) == 1, "You can select only one classifier at one time"
        return str(self.selected_classifier[0])

    def __init_classifier(self):
        """
        Initialize classifier class
        :return Initialized classifier
        """
        classifier = None
        # Initialize chosen classifier
        if self.selected_classifier == "knn":
            classifier = kNN(self.k)
        elif self.selected_classifier == "mlp":
            classifier = MLP(self.validation_rate)
        assert classifier is not None, "No classifier selected"
        return classifier

    def load_model(self, input_model_file_name: str):
        """
        Load trained model
        :param  input_model_file_name: model file name
        :return trained model
        """
        assert os.path.exists(input_model_file_name), "Selected model does not exist"
        return self.classifier.load_model(input_model_file_name)

    def save_model(self, model, output_directory: str):
        """
        Save trained model
        :param  model: trained model
        :param  output_directory: model file name
        :return trained model
        """
        # Make output directory if it does not exist
        if os.path.exists(output_directory) is False:
            os.mkdir(output_directory)

        # Make model file name
        model_filename = "{}.h5".format(self.selected_classifier)

        # Save model
        self.classifier.save_model(model, os.path.join(output_directory, model_filename))

    def training(self, train_data, train_label):
        """
        Training with train data set
        :param   train_data:  training data
        :param   train_label: test data
        :return  model: trained   model
        """
        # Train model
        return self.classifier.training(train_data, train_label)

    def predict(self, model, test_data, test_label) -> float:
        """
        Make predictions with a given model to test data set
        :param  model: trained model
        :param  test_data: training data
        :param  test_label: test data
        :return prediction score
        """
        return self.classifier.predict(model, test_data, test_label)

    def show_history(self, model_training):

        # Create the plot for training history
        # Accuracy
        plt.plot(model_training.history['acc'])
        plt.plot(model_training.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # Loss
        plt.plot(model_training.history['loss'])
        plt.plot(model_training.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
