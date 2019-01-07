#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:46:59 2017

@author: akihiro
"""
#==============================================================================
# ExtractAll.py
# Program author: Akihiro Inui
# Extract features from all genres
#==============================================================================

#==============================================================================
# 0. Import libraries and make a path
#==============================================================================
from progressbar import ProgressBar, Percentage, Bar
import time
import pandas as pd
import numpy as np
import os
from ml import FeatureExtraction
from data_process import normalise, datacleaner
from data_process.standardise import standardise
file_path = "[Your Working Directory]"
os.chdir(file_path)
#==============================================================================
# 1. Custom User Input
#==============================================================================
def ExtractAll(custom):
    # 0:Off,1:On
    custom = 0
    
    if custom == 1:
        
        m = int(input('MFCC:'))
        c = int(input('Centroid:'))
        r = int(input('Rolloff:'))
        f = int(input('Flux:'))
        z = int(input('Zero Crossing Rate:'))
        o = int(input('OSC:'))
        L = int(input('Low Energy:'))
        OM = int(input('OMSC:'))
        MM = int(input('MSFM:'))
        MC = int(input('MSCM:'))
    
        #Feature selection 
        feature = [m,c,r,f,z,o,L,OM,MM,MC]
    
    feature = np.ones(10)
    
    # Feature dimension information
    Finfo = {'MFCC':15,'Centroid':1,'Rolloff':1,'Flux':1,'ZCR':1,'OSC':18,\
             'LowEnergy':1,'OMSC':10,'MSFM':10,'MSCM':10}
    
    # Dimension
    featuredim = sum(Finfo.values())
#==============================================================================
# 2. Read genre names and prepare labels
#==============================================================================
    # Move into target folder and read genre names
    genres = os.listdir("GTZAN")
    if ".DS_Store" in genres:
        genres.remove(".DS_Store")
    genres = sorted(genres)
    os.chdir("GTZAN")
#==============================================================================
# 3. Extract genre
#==============================================================================
    # For each genre
    labelnum = 0
    genrenum = 0
    for genre in genres:
        # Print status
        print("\n" + "Extracting {}".format(genre) +"\n" )
        
        # read filenames in genre
        files = os.listdir(genre)
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        files = sorted(files)
        
        # Move to genre
        os.chdir(genre)
        
        # Create label vector
        labelnum += 1
        label = labelnum*np.ones(len(files))
        label = np.array([label]).T
#==============================================================================
# 4. Extract file
#==============================================================================
        # For each audio file
        filenum = 0
        # Progress bar
        bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=100).start()
        for file in files:
            # Print status
            #print("Target file is %s" %file)
            
            # Feature extraction to one file
            f = FeatureExtraction.FeatureExtraction(file)
            featurelist = f.features()
            featurefile = []
            
            # Vectorise feature
            for i in featurelist:
                featurefile = np.append(featurefile,i)
                
            # Connect to previous feature vectors
            if filenum == 0:
                featuregenre = np.zeros(len(featurefile))
            featuregenre = np.vstack([featuregenre,featurefile])
            
            # End of file #
            bar.update(filenum+1)
            time.sleep(0.01)
            
            # Update counter
            filenum += 1
            
        # End status bar
        bar.finish()
        
        # Transpose feature matrix and add label
        featuregenre = featuregenre[1:]
        featuregenre = np.hstack([featuregenre,label])
        
        # Connect feature matrices
        if genrenum == 0:
            FeatureMat = np.zeros(len(featurefile)+1)
        FeatureMat = np.vstack([FeatureMat,featuregenre])
        # End of genre #
        genrenum += 1
        os.chdir("../")
#==============================================================================
# 5. Data cleaning/Normalisation or Standardisation
#==============================================================================    
    # Extract first row
    FeatureMat = FeatureMat[1:]
    
    # Clean data    
    FeatureMat = datacleaner.datacleaner(FeatureMat, 0, 0)
    Data = pd.DataFrame(FeatureMat)
    
    # Separation to data and label
    Label = Data.iloc[:,Data.shape[1]-1]
    
    # Normalisation or Standardisation 
    Rescale = 0
    
    # Normalise
    if Rescale == 0: 
        # Normalise FeatureA
        Content = Data.iloc[:,0:Data.shape[1]-1]
        Content = pd.DataFrame(normalise(Content.values, 0))
        
    # Standardise
    elif Rescale == 1:
        # Standardise FeatureA
        Content = Data.iloc[:,0:Data.shape[1]-1]
        Content = pd.DataFrame(standardise(Content.values,0))
        
    # Combine data and label
    Data = pd.concat([Content, Label], axis=1)
    os.chdir(file_path)
    Data.to_csv("Data.csv")
    
    return Data
