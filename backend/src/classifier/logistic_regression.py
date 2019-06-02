#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17 May 2019
@author: Akihiro Inui
"""

import torch
import numpy as np
from torch import nn, optim
import pandas as pd
import torch.nn.functional as F


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
        Training for Gated Recurrent Unit
        :param  train_data: training data
        :param  train_label: train label
        :param  visualize: True/False to visualize training history
        :return model: trained model
        """
        # Convert training data
        train_data = torch.tensor(np.array(train_data), dtype=torch.float32)

        # One hot encode
        onehot_train_label = torch.tensor(np.array(train_label), dtype=torch.long)
        #onehot_train_label = torch.tensor(np.array(pd.get_dummies(train_label)), dtype=torch.long)

        # Define the model
        model = nn.Linear(train_data.shape[1], self.num_classes)

        # Softmax Cross Entropy
        loss_fn = nn.CrossEntropyLoss()

        # SGD
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # To log losses
        losses = []
        for epoch in range(100):
            # Delete gradient value calculated in previous epoch
            optimizer.zero_grad()

            # Make prediction
            predicted_label = model(train_data)

            # Calculate MSE loss
            a = torch.max(predicted_label, 1)[1]
            loss = loss_fn(torch.max(predicted_label, 1)[1], onehot_train_label)
            loss.backward()

            # Update gradient
            optimizer.step()

            # Log loss
            losses.append(loss.item())

        return model

    def load_model(self, model_file_name: str):
        """
        Load trained model
        :param  model_file_name: name of model file to load
        :return model: trained model
        """
        # Load model if it exists
        assert os.path.exists(model_file_name), "Given model file does not exist"
        return load_model(model_file_name)

    def save_model(self, model, output_directory: str):
        """
        Save model
        :param  model: trained model
        :param  output_directory: output directory path
        """
        model.save(os.path.join(output_directory, "gru.h5"))

    def test(self, model, test_data, test_label):
        """
        Make a test for the given dataset
        :param  model: trained model
        :param  test_data: test data
        :param  test_label: test label
        :return result of test
        """
        # One hot encode
        onehot_test_label = pd.get_dummies(test_label)
        # Make predictions and output result
        return model.evaluate(test_data, onehot_test_label)[1]

    def predict(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data without label
        :return prediction array with probability
        """
        # Make prediction to the target data
        return model.predict(target_data)
