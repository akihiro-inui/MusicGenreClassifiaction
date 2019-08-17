#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sleepy midnight in June
@author: Akihiro Inui
"""

from __future__ import print_function
import onnx
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

    def __init__(self, num_classes: int, image_size: int):
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

        # Transformer
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5], [0.5])
        ])

    def torch_train(self, epoch_num, train_loader, validation_loader, visualize_history=True):
        """
        Training/Validation for ResNet
        :param epoch_num: Number of total epoch
        :param train_loader: Training Data loader
        :param validation_loader: Validation Data loader
        :param visualize_history: Set True to visualize loss and accuracy history
        """

        # Keep history
        training_loss_history = []
        training_accuracy_history = []
        validation_loss_history = []
        validation_accuracy_history = []

        # For every epoch
        for epoch in range(1, epoch_num + 1):

            # Set model to training mode
            self.model.train()

            # Initialize parameters
            training_loss_history_epoch = []
            training_accuracy_history_epoch = []

            # For every batch from train loader
            for batch_idx, data in enumerate(train_loader):
                # Split Data/Label
                image, label = data
                image, label = Variable(image), Variable(label)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Make prediction
                output = self.model(image)

                # Calculate loss, store it as history
                loss = self.loss_function(output, label)
                training_loss_history_epoch.append(loss)

                # Perform a backward pass, and update the weights.
                loss.backward()
                self.optimizer.step()

                # Calculate training accuracy, store it as history
                _, prediction = output.max(1)
                accuracy = (label == prediction).float().sum().item() / len(label)
                training_accuracy_history_epoch.append(accuracy)

                # Print state of training
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}\tTraining Accuracy {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data, accuracy))

            # Training loss and accuracy (Take average performance of this epoch)
            training_loss_history.append(sum(training_loss_history_epoch) / len(training_loss_history_epoch))
            training_accuracy_history.append(
                sum(training_accuracy_history_epoch) / len(training_accuracy_history_epoch))

            # Validation loss and accuracy (Take average performance of this epoch)
            validation_accuracy_history_epoch, validation_loss_history_epoch = self.torch_validation(validation_loader)
            validation_accuracy_history.append(
                sum(validation_accuracy_history_epoch) / len(validation_accuracy_history_epoch))
            validation_loss_history.append(sum(validation_loss_history_epoch) / len(validation_loss_history_epoch))

        # Show loss and accuracy history
        if visualize_history is True:
            self.torch_visualize(training_loss_history, validation_loss_history, training_accuracy_history,
                                 validation_accuracy_history)

    def torch_validation(self, validation_loader):
        """
        Validation for ResNet
        :param validation_loader: Validation Dataset loader
        :return validation_accuracy_history_epoch: Validation accuracy in list
        :return validation_loss_history_epoch: Validation loss in list
        """
        # Set model to evaluation mode
        self.model.eval()

        # To save history
        validation_accuracy_history_epoch = []
        validation_loss_history_epoch = []
        for batch_idx, data in enumerate(validation_loader):
            # Split validation data into image and label
            image, label = data
            image, label = Variable(image), Variable(label)

            # Make prediction
            output = self.model(image)

            # Calculate loss, store it as history
            loss = self.loss_function(output, label)
            validation_loss_history_epoch.append(loss)

            # Predict class
            with torch.no_grad():
                _, prediction = self.model(image).max(1)

            # Append to list
            validation_accuracy_history_epoch.append((label == prediction).float().sum().item() / len(label))

        return validation_accuracy_history_epoch, validation_loss_history_epoch

    def torch_test(self, test_loader):
        """
        Test for ResNet
        :param test_loader: Test Dataset loader
        """
        # Init param
        test_loss = 0
        correct = 0

        # Set model to evaluation mode
        self.model.eval()
        for (image, label) in test_loader:
            # Split test dataset into data and label
            image, target = Variable(image.float(), volatile=True), Variable(label)

            # Make prediction
            output = self.model(image)

            # Sum up batch loss
            test_loss += self.loss_function(output, label).data

            # Get the index of the max log-probability (predicted class)
            pred = output.data.max(1, keepdim=True)[1]

            # Count correct predictions
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Calculate test loss and accuracy
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def torch_predict(self, image_data):
        """
        Make prediction to one image file.
        Feed into the model
        :param image_data: PIL loaded image data
        """
        # Set to evaluation mode
        self.model.eval()

        # Pass the image through our model with Softmax to get probability
        output = self.model(image_data)
        output = F.softmax(output, dim=1)

        # Make prediction and get the highest confident class
        confidence, highest_class = torch.topk(output, 1)
        confidence = confidence.item()
        highest_class = highest_class.item()
        return confidence, highest_class

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

