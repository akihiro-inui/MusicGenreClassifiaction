#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sleepy midnight in June
@author: Akihiro Inui
"""

from __future__ import print_function
import onnx
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from onnx_tf.backend import prepare
from onnx_coreml import convert
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras


class ResNet:
    """
    Based on ResNet using transfer learning by adding the final fully connected layer.
    How to use this model outside of this class.
    1. model = ResNet(num_classes, image_size)
    2. model.torch_load(param_file_oath)
    3. image_data = model.transform(Image.open(image_file)) , image_data = image_data.unsqueeze(0)
    4. confidence, predicted_class = model.torch_predict(image_data)
    """

    def __init__(self, validation_rate, num_classes):
        # Rate for validation
        self.validation_rate = validation_rate

        # Number of classes
        self.num_classes = num_classes

        # Use pre-trained ResNet
        self.model = models.resnet50(pretrained=True)

        # Exclude all parameters to be updated (Fix parameters for ResNet excluding the final fully connected layer)
        for param in self.model.parameters():
            param.requires_grad = False

        # Add final layer to ResNet (n classes classification)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Define Loss function
        self.loss_function = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)

    def training(self, train_data, train_label, visualize=True):
        """
        Training for ResNet
        :param  train_data: training data
        :param  train_label: train label
        :param  visualize: True/False to visualize training history
        :return model: trained model
        """
        train_samples_num = train_data.shape[0]
        data_width = train_data.shape[1]
        data_height = train_data.shape[2]
        train_data = train_data.reshape((train_samples_num, data_width, data_height, 1))
        # Convert training data
        train_data = torch.tensor(train_data, dtype=torch.float32)

        # One hot encode
        onehot_train_label = torch.tensor(train_label, dtype=torch.long)

        # For every epoch
        losses = []
        for epoch in range(300):
            # Set model to training mode
            self.model.train()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Make prediction
            predicted_label = self.model(train_data)

            # Calculate loss
            loss = self.loss_function(predicted_label, onehot_train_label)
            loss.backward()

            # Update gradient
            self.optimizer.step()

            # Log loss
            losses.append(loss.item())

        # Visualize losses
        if visualize is True:
            plt.plot(losses)
            plt.show()
        return self.model

    def test(self, model, test_data, test_label, is_classification=True):
        """
        Make a test for the given dataset
        :param  model: trained model
        :param  test_data: test data
        :param  test_label: test label
        :param  is_classification: Bool
        :return result of test
        """
        # Reshape data
        test_samples_num = test_data.shape[0]
        data_width = test_data.shape[1]
        data_height = test_data.shape[2]
        test_data.reshape((test_samples_num, data_width, data_height, 1))

        # One hot encode
        onehot_test_label = torch.tensor(test_label, dtype=torch.long)

        # Make prediction
        prediction = model(torch.tensor(test_data), dtype=torch.float32)

        _, predicted_classes = torch.max(prediction, 1)

        # Treat max value as predicted class
        predicted_classes = torch.max(prediction, 1)[1]

        return (predicted_classes == onehot_test_label).sum().item()/len(test_label)

    def predict(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data without label
        :return prediction array with probability
        """
        # Make prediction to the target data
        return model(torch.tensor(np.array(target_data), dtype=torch.float32))

    def torch_save(self, param_file_path: str):
        """
        Save weights with .pth extension
        :param param_file_path: Parameter file name to be saved
        """
        # Save weight of the model
        torch.save(self.model.state_dict(), param_file_path)

    def torch_load(self, param_file_path: str):
        """
        Load weights with .pth extension
        :param param_file_path: Parameter file name to be loaded
        """
        # Save weight of the model
        self.model.load_state_dict(torch.load(param_file_path))

    @staticmethod
    def torch_visualize(training_loss_history: list, validation_loss_history: list, training_accuracy_history: list,
                        validation_accuracy_history: list):
        """
        Visualize loss and history of training and validation
        :param training_loss_history: Loss of training in list
        :param validation_loss_history: Loss of validation in list
        :param training_accuracy_history: Accuracy of training in list
        :param validation_accuracy_history: Accuracy of validation in list
        """
        plt.plot(training_loss_history)
        plt.plot(validation_loss_history)
        plt.title('Loss history')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.plot(training_accuracy_history)
        plt.plot(validation_accuracy_history)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def export_onnx(self, onnx_model_file_name="resnet.onnx"):
        """
        Export to ONNX model format
        """
        # Input to the model (sample_batch_siz, channel, height, width)
        input_shape = torch.randn(1, 3, 224, 224, requires_grad=True)

        # Export ONNX model
        onnx_model = torch.onnx.export(self.model,  # model being run
                                       input_shape,  # model input (or a tuple for multiple inputs)
                                       onnx_model_file_name,
                                       # where to save the model (can be a file or file-like object)
                                       export_params=True,
                                       input_names=['input'], output_names=['output'])  # store the trained parameter weights inside the model file
        return onnx_model

    def export_coreml(self, onnx_file_name='resnet.onnx', coreml_model_file_name="resnet.mlmodel"):
        """
        Export to coreml model format
        """
        # Export the model
        onnx_model = onnx.load(onnx_file_name)

        # Export model to coreml model
        coreml_model = convert(onnx_model)

        # Export CoreML model
        coreml_model.save(coreml_model_file_name)
        return coreml_model

    def export_h5(self, h5_model_file_name="resnet.h5"):
        """
        Export to h5 model format
        """
        # Input to the model (sample_batch_siz, channel, height, width)
        input_shape = torch.randn(1, 3, 224, 224)

        # Export model to h5 model
        k_model = pytorch_to_keras(self.model, input_shape, [(3, None, None,)], verbose=True)
        k_model.save(h5_model_file_name)

    def export_pb(self, onnx_file_name='resnet.onnx', tf_model_file_name='resnet.pb'):
        """
        Export to pb format (tensorflow)
        """

        # Convert to ONNX once
        onnx_model = onnx.load(onnx_file_name)
        onnx.checker.check_model(onnx_model)

        # Receive with tf
        tf_rep = prepare(onnx_model)

        # Print out tensors and placeholders in model (helpful during inference in TensorFlow)
        #print(tf_rep.tensor_dict)

        # Export model as .pb file
        tf_rep.export_graph(tf_model_file_name)

