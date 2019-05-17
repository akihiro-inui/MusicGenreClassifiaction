#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape


class GatedRecurrentUnit:
    """
    Gated Recurrent Unit
    """
    def __init__(self, validation_rate, num_classes):
        """
        Init
        :param  validation_rate: validation rate
        :param  num_classes: number of classes
        :return model: trained model
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
        # One hot encode
        onehot_train_label = pd.get_dummies(train_label)

        # Time series length, Feature length
        input_shape = (train_data.shape[1], train_data.shape[2])

        # Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=2)

        # Set up the model: model
        # n_hidden = 16
        epochs = 2
        batch_size = 10
        model = Sequential()

        model.add(GRU(50, input_shape=input_shape, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
        model.add(GRU(self.num_classes, return_sequences=False))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_graph = model.fit(train_data, onehot_train_label,
                  batch_size=batch_size, epochs=epochs, validation_split=self.validation_rate)

        # Visualize training history
        if visualize is True:
            # Create the plot for training history
            # Accuracy
            plt.plot(model_graph.history['acc'])
            plt.plot(model_graph.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            # Loss
            plt.plot(model_graph.history['loss'])
            plt.plot(model_graph.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
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
