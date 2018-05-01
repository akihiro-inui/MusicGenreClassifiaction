#==============================================================================
# normalise.py
# Program author: Akihiro Inui
# Compute Normalisation to Data Matrix 
#==============================================================================
import numpy as np
#==============================================================================
# Define Normalisation
#==============================================================================
# Data = Input Data
# type = 0, Normalise by column
#      = 1, Normalise by row
def normalise(Data,type):
    
    # Check dimension
    Dim = Data.ndim
    
    # Create an empty matrix
    NormData = np.zeros((Data.shape[0],Data.shape[1]))
    
    # Case of vector input
    if Dim == 1:
        for n in range(0,Data.shape[0]):
            Data = (Data-min(Data))/(max(Data)-min(Data))
            
    # Case of matrix input
    else:
        
    # Normalise by column
        if type == 0:
            for n in range(0,Data.shape[1]):
                DataColumn = Data[:,n]
                DataColumn = (DataColumn-min(DataColumn))/(max(DataColumn)-min(DataColumn))
                NormData[:,n] = DataColumn
                
    # Normalise by row
        else:
            for m in range(0,Data.shape[0]):
                DataRow = Data[m,:]
                DataRow = (DataRow-min(DataRow))/(max(DataRow)-min(DataRow))
                NormData[m,:] = DataRow
                
    return NormData