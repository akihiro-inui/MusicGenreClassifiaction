#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sleepy midnight in Aug
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
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import tqdm

class FlattenLayer(nn.Module):
    def forward(self, inputs):
        sizes = inputs.size()
        return inputs.view(sizes[0], -1)


class CNN:
    def __init__(self, validation_rate, num_classes):
        self.validation_rate = validation_rate
        self.num_classes = num_classes

        conv = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            FlattenLayer()
        )

        mlp = nn.Sequential(
            nn.Linear(10816, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200, num_classes),
        )

        self.model = nn.Sequential(conv, mlp)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.CrossEntropyLoss()

    def training(self, train_loader, validation_loader, visualize=True):
        # To log losses
        loss_history = []
        accuracy_history = []
        num = 0

        for epoch in range(1, 30):
            # Iteration for batch
            batch_accuracy = 0
            batch_loss = 0
            for batch_idx, (data, label) in tqdm.tqdm(enumerate(train_loader)):
                # Define hardware
                data = data.to("cpu")
                label = label.to("cpu")

                # Forward data and calculate loss
                data = data.unsqueeze(1)  # Make 3D array to 4D
                output = self.model(data)
                _, prediction = output.max(1)
                loss = self.loss_function(output, label)

                # Delete gradient value calculated in previous epoch
                self.optimizer.zero_grad()

                # Log loss
                batch_loss += loss.item()
                loss.backward()

                # Update optimizer
                self.optimizer.step()

                num += len(data)
                batch_accuracy += (label == prediction).float().sum().item()

            # Append accuracy and loss
            accuracy_history.append(batch_accuracy/num)
            loss_history.append(batch_loss/num)

            # Print training status
            print(loss_history[-1], accuracy_history[-1], flush=True)

        if visualize is True:
            plt.plot(loss_history)
            # plt.plot(validation_loss_history)
            plt.title('Loss history')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        return self

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            # Split test dataset into data and label
            data, target = Variable(data, volatile=True), Variable(target)

            # Make prediction
            output = self.model(data)

            # Sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).data

            # Get the index of the max log-probability (predicted class)
            pred = output.data.max(1, keepdim=True)[1]

            # Count correct predictions
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def predict(self, model, target_data):
        """
        Make prediction to a given target data and return the prediction result with accuracy for each sample
        :param  model: trained model
        :param  target_data: target data without label
        :return prediction array with probability
        """
        # Make prediction to the target data
        return model(torch.tensor(np.array(target_data), dtype=torch.float32))

    def save_model(self, param_file_path: str):
        """
        Save weights with .pth extension
        :param param_file_path: Parameter file name to be saved
        """
        # Save weight of the model
        torch.save(self.state_dict(), param_file_path)

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

