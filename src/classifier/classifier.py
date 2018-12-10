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
        self.selected_classifier = [classifier for classifier, switch in self.classifier_selection_dict.items() if switch == "True"]
        assert len(self.selected_classifier) == 1, "You can select only one classifier at one time"
        return str(self.selected_classifier[0])

    def __init_classifier(self):
        """
        Initialize classifier class
        :return Initialized classifier
        """
        # Initialize chosen classifier
        if self.selected_classifier == "knn":
            classifier = kNN(self.k)
        elif self.selected_classifier == "mlp":
            classifier = MLP(self.validation_rate)
        return classifier

    def load_model(self, model_file_name: str):
        """
        Load trained model
        :param  model_file_name: model file name
        :return trained model
        """
        return self.classifier.load_model(model_file_name)

    def training(self, train_data, train_label, model_file_name: str=None):
        """
        Execute training
        Case A:  Load trained model
        Case B:  Train model
        :param   train_data:  training data
        :param   train_label: test data
        :param   model_file_name: model file name
        :return  model: trained   model
        """
        # Start training model if it does not receive model file name (Case B)
        if model_file_name is None:
            # Train model
            model = self.classifier.training(train_data, train_label)
            # Save model
            self.classifier.save_model(model, "../model/{0}.h5".format(self.selected_classifier))
            return model
        else:
            # Pass trained model file if it exists (Case A)
            if os.path.exists(model_file_name):
                return self.load_model(model_file_name)
            # Train model if given model file does not exist (Case B)
            else:
                print("Given model file does not exist, start training")
                # Train model
                model = self.classifier.training(train_data, train_label)
                # Save model
                self.classifier.save_model(model, "../model/{0}.h5".format(self.selected_classifier))
                return model

    def predict(self, model, test_data, test_label) -> float:
        """
        Execute prediction
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
