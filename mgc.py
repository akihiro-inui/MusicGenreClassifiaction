#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:12:51 2018

@author: akihiro
"""
#==============================================================================
# mgc.py
# Program author: Akihiro Inui
# Implement Music Genre Classification
#==============================================================================
import ExtractAll
import Classification as cl
import os
#==============================================================================
# Path to Data
#==============================================================================
path = "/Users/akihiro/Documents/Codes/Github/MusicGenreClassification-Python-"
os.chdir(path)
#==============================================================================
# 1. Feature Extraction to Data
#==============================================================================
#Data = ExtractAll.ExtractAll(0)
#==============================================================================
# 2. Classification
#==============================================================================
#  Train rate
train = 0.7
     
# Compute classification itr times
print("\n Start Classifying \n")

# Load data
Target = cl.Classification("Data.csv")
    
# Make dataset with order randomisation
Dataset = Target.mkdataset(train)
    
# Classify (Comment out if it's not neccesary)

#accuracykNN = Target.kNN(Dataset,5)
accuracyMLP = Target.mlp(Dataset)
#accuracyCNN = Target.cnn("Featuremap")

#==============================================================================
# 3. Result
#==============================================================================

# Print out the result
#print("Accuracy with {0} is {1}%".format("kNN",accuracykNN))
print("Accuracy with {0} is {1}%".format("MLP",accuracyMLP))
#print("Accuracy with {0} is {1}%".format("CNN",accuracyCNN))
