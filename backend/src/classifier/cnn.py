#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018
Convolutional Neural Network
@author: Akihiro Inui
"""

import os
import pickle
import pandas as pd
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, validation_rate: float,  num_classes: int):
        """
        Init
        :param  validation_rate: validation rate
        :param  num_classes: number of classes
        :return model: trained model
        """
        self.validation_rate = validation_rate
        self.num_classes = num_classes

    def training(self, train_data, train_label, visualize=False):
        """
        Training for CNN
        :param  train_data: training data
        :param  train_label: train label
        :param  visualize: visualize training history
        :return model: trained model
        """

        # One hot encode label
        onehot_train_label = pd.get_dummies(train_label)

        # Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=2)

        # Time series length, Feature length
        input_shape = (train_data.shape[1], train_data.shape[2])

        # Set up the model: model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["accuracy"])

        # Fit the model
        model_graph = model.fit(train_data, onehot_train_label, callbacks=[early_stopping_monitor], epochs=5, shuffle=False, validation_split=self.validation_rate)

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
        :param  output_directory: output directory
        """
        model.save(os.path.join(output_directory, "cnn.h5"))

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
        return model.predict_proba(target_data)
