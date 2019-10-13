#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018
@author: Akihiro Inui
"""

# Import libraries/modules
import os
import pickle
import torch
import cloudpickle
import matplotlib.pyplot as plt
from backend.src.classifier.kNN import kNN
from backend.src.classifier.cnn import CNN
from backend.src.classifier.mlp import MultiLayerPerceptron as MLP
from backend.src.classifier.resnet import ResNet
from backend.src.classifier.rnn import CustomRNN
from backend.src.data_process.data_process import DataProcess
from backend.src.classifier.logistic_regression import LogisticRegression
from backend.src.common.config_reader import ConfigReader
from keras.models import load_model


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
        self.num_classes = self.cfg.num_classes
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
        elif self.selected_classifier == "logistic_regression":
            classifier = LogisticRegression(self.validation_rate, self.num_classes)
        elif self.selected_classifier == "mlp":
            classifier = MLP(self.validation_rate, self.num_classes)
        elif self.selected_classifier == "cnn":
            classifier = CNN(self.validation_rate, self.num_classes)
        elif self.selected_classifier == "resnet":
           classifier = ResNet(self.validation_rate, self.num_classes)
        elif self.selected_classifier == "rnn":
            classifier = CustomRNN(self.validation_rate, self.num_classes)
        assert classifier is not None, "No classifier selected. Please select one"
        return classifier

    def make_dataset_loader(self, train_data_with_label, test_data_with_label, validation_rate):
        """
        Dataset loader for Torch
        :return train loader, test loader
        """
        # Make Dataset loader
        train_loader, validation_loader, test_loader = DataProcess.torch_data_loader(train_data_with_label, test_data_with_label, validation_rate)
        return train_loader, validation_loader, test_loader

    def load_model(self, input_model_file_name: str):
        """
        Load trained model
        :param  input_model_file_name: model file name
        :return trained model
        """
        assert os.path.exists(input_model_file_name), "Selected model does not exist"
        # Load/Unpickle model
        if input_model_file_name.endswith(".pickle"):
            with open(input_model_file_name, mode='rb') as fp:
                loaded_model = pickle.load(fp)
        elif input_model_file_name.endswith(".prm"):
            loaded_model = torch.load(input_model_file_name)
            loaded_model.eval()
        elif input_model_file_name.endswith(".pkl"):
            with open(input_model_file_name, 'rb') as f:
                loaded_model = cloudpickle.load(f)
        else:
            loaded_model = load_model(input_model_file_name)
        return loaded_model

    def save_model(self, model, output_directory: str):
        """
        Pickle trained model
        :param  model: trained model
        :param  output_directory: pickled model file name
        """
        # Make output directory if it does not exist
        if os.path.exists(output_directory) is False:
            os.mkdir(output_directory)
        self.classifier.save_model(model, output_directory)

    def training(self, train_data, train_label, visualize=None):
        """
        Training with train data set
        :param   train_data:  training data
        :param   train_label: training data
        :param   visualize: True/False to visualize training history
        :return  model: trained   model
        """
        print("Train Started")
        if self.selected_classifier == "cnn" or self.selected_classifier == "resnet" or self.selected_classifier == "rnn":
            # Make Torch dataset loader for train
            train_loader, validation_loader = DataProcess.torch_train_data_loader(train_data, train_label, self.validation_rate)
            # Train model
            return self.classifier.training(train_loader, validation_loader, visualize=visualize)
        else:
            # Train model
            return self.classifier.training(train_data, train_label, visualize)

    def test(self, model, test_data, test_label) -> float:
        """
        Make predictions and output the result from a given model to test data set
        :param  model: trained model
        :param  test_data: test data
        :param  test_label: test data
        :return Over all test score (accuracy)
        """
        print("Test Started")
        if self.selected_classifier == "cnn" or self.selected_classifier == "resnet" or self.selected_classifier == "rnn":
            # Make Torch dataset loader for test
            test_loader = DataProcess.torch_test_data_loader(test_data, test_label)
            # Test model performance
            return self.classifier.test(model, test_loader)
        else:
            # Test model performance
            return self.classifier.test(model, test_data, test_label)

    def predict(self, model, target_data) -> list:
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data
        :return prediction result with accuracy
        """
        return self.classifier.predict(model, target_data)

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
