#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:14:28 2018

@author: akihiro inui
"""

import os
import pickle
import pandas as pd
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model


class MLP:
    """
    Multi Layer Perceptron
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
        Training for Multi layer perceptron
        :param  train_data: training data
        :param  train_label: train label
        :return model: trained model
        """
        # One hot encode
        onehot_train_label = pd.get_dummies(train_label)

        # Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=2)

        # Set up the model: model
        model = Sequential()

        # Learning Rate
        LearningRate = 0.001

        # Add the first layer
        model.add(Dense(200, activation='relu', input_shape=(train_data.shape[1],)))

        # Add the second layer
        model.add(Dense(150, activation='relu'))

        # Add the second layer
        model.add(Dense(150, activation='relu'))

        # Add the output layer
        model.add(Dense(self.num_classes))

        # Create optimizer
        my_optimizer = SGD(lr=LearningRate)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        # model.compile(optimizer=my_optimizer, loss='mean_squared_error',metrics=['accuracy'])
        # model.compile(optimizer=my_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

        # Fit the model
        model.fit(train_data, onehot_train_label, callbacks=[early_stopping_monitor],
                  epochs=50, shuffle=False, validation_split=self.validation_rate)

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
        :param  output_directory: name of model file to load
        """
        model.save(os.path.join(output_directory, "mlp.h5"))

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


    def show_history(self, model):
        # Load model
        my_model.summary()

        ## Create the plot for training history
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

        # Evaluation with test data
        score = my_model.evaluate(TestData, TestLabel)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])
        score = 100 * score[1]

        return score
