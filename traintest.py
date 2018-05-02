#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:15:58 2017

@author: akihiro
"""

#==============================================================================
# traintest.py
# Program author: Akihiro Inui
# separate image data into train and test
# Mode1(Data_Type=0), move images into Train/Test folder
# Mode2 (Data_Type=1), separate a big feature matrix into Train/Test folder
#==============================================================================
#==============================================================================
# 0. Import libraries and functions
#==============================================================================
import os
import shutil
import random
#==============================================================================
# 1. Read data 
#==============================================================================
# Input target data folder
Data_Folder = str(input('Target Folder?:'))
Data_Type = int(input('Data Type? (Image:0,Vector:1)'))
Data_Path = "[Your Working Directory]" + "/" + Data_Folder
Train_Rate = float(input('Train Rate?:'))

# Destination folder names
Train_Folder = Data_Folder + '_Train'
Test_Folder = Data_Folder + '_Test'

os.mkdir(Train_Folder)
os.mkdir(Test_Folder)

# Read folder names
Genres = os.listdir(Data_Path)
if '.DS_Store' in Genres:
    os.remove(Data_Path + '/.DS_Store')
    Genres = os.listdir(Data_Path)

# Make Train folders
os.chdir(Train_Folder)
for j in range(len(Genres)):
    os.mkdir(Genres[j])
os.chdir('../')

# Make Test folders
os.chdir(Test_Folder)
for j in range(len(Genres)):
    os.mkdir(Genres[j])
os.chdir('../')

# Data separation for each genre
for i in range(len(Genres)):
    os.chdir(Data_Path)
    # List of file names
    List = os.listdir(Genres[i])
    os.chdir(Genres[i])
    file_num = len(List)
    train_num = round(file_num*Train_Rate)
    test_num = int(len(List)-train_num)
    
    # Shuffle files
    random.shuffle(List)
    
    # Train data/Test data
    train_files = List[0:train_num]
    test_files = List[train_num:]
    
    # Copy Train data
    Train_Folder_Genre = "/Users/akihiro/Documents/Codes/Github/MusicGenreClassification-Python-" + '/' +Train_Folder + '/' + Genres[i]
    for tr in range(train_num):
        File = train_files[tr]
        shutil.copy2(File,Train_Folder_Genre)
    
    # Copy Test data
    Test_Folder_Genre = "/Users/akihiro/Documents/Codes/Github/MusicGenreClassification-Python-" + '/' +Test_Folder + '/' + Genres[i]
    for te in range(test_num):
        File = test_files[te]
        shutil.copy2(File,Test_Folder_Genre)
    
