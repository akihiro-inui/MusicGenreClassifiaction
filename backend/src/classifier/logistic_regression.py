#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17 May 2019
@author: Akihiro Inui
"""

import os
import torch
import numpy as np
import cloudpickle
from torch import nn, optim
import matplotlib.pyplot as plt


class TorchLinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class LogisticRegression:
    """
    Logistic Regression
    """
    def __init__(self, validation_rate, num_classes):
        """
        Init
        :param  validation_rate: validation rate
        :param  num_classes: number of classes
        """
        self.validation_rate = validation_rate
        self.num_classes = num_classes

    def training(self, train_data, train_label, visualize=None):
        """
        Training for Logistic Regression
        :param  train_data: training data
        :param  train_label: train label
        :param  visualize: True/False to visualize training history
        :return model: trained model
        """
        # Convert training data
        train_data = torch.tensor(np.array(train_data), dtype=torch.float32)

        # One hot encode
        onehot_train_label = torch.tensor(np.array(train_label), dtype=torch.long)

        # Define the model
        model = TorchLinearRegression(train_data.shape[1], self.num_classes)

        # Cross Entropy
        loss_fn = nn.CrossEntropyLoss()

        # SGD
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # To log losses
        losses = []
        for epoch in range(100000):
            # Delete gradient value calculated in previous epoch
            optimizer.zero_grad()

            # Make prediction
            predicted_label = model(train_data)

            # Calculate MSE loss
            loss = loss_fn(predicted_label, onehot_train_label)
            loss.backward()

            # Update gradient
            optimizer.step()

            # Log loss
            losses.append(loss.item())

        # Visualize losses
        if visualize is True:
            plt.plot(losses)
            plt.show()

        return model

    def load_model(self, model_file_name: str):
        """
        Load trained model
        :param  model_file_name: name of model file to load
        :return model: trained model
        """
        # Load model if it exists
        assert os.path.exists(model_file_name), "Given model file does not exist"
        return torch.load(model_file_name, map_location="cpu")

    def save_model(self, model, output_directory: str):
        """
        Save model
        :param  model: trained model
        :param  output_directory: output directory path
        """
        with open(os.path.join(output_directory, "logistic_regression.pkl"), 'wb') as f:
            cloudpickle.dump(model, f)
        # torch.save(model.state_dict(), os.path.join(output_directory, "logistic_regression.prm"), pickle_protocol=4)

    def test(self, model, test_data, test_label, is_classification=True):
        """
        Make a test for the given dataset
        :param  model: trained model
        :param  test_data: test data
        :param  test_label: test label
        :param  is_classification: Bool
        :return result of test
        """
        # One hot encode
        onehot_test_label = torch.tensor(np.array(test_label), dtype=torch.long)

        # Make prediction
        prediction = model(torch.tensor(np.array(test_data), dtype=torch.float32))

        _, predicted_classes = torch.max(prediction, 1)

        # Treat max value as predicted class
        predicted_classes = torch.max(prediction, 1)[1]

        return (predicted_classes == onehot_test_label).sum().item()/len(test_label) * 100

    def predict(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data without label
        :return prediction array with probability
        """
        # Make prediction to the target data
        output = model(torch.tensor(target_data, dtype=torch.float32))
        return np.array(output.data)
