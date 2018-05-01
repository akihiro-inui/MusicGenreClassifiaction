#==============================================================================
# standardise.py
# Program author: Akihiro Inui
# Compute Normalisation to Data Matrix 
#==============================================================================
import numpy as np
#==============================================================================
# Define Normalisation
#==============================================================================
# Data = Input Data
# type = 0, Standardise by column
#      = 1, Standardise by row
def standardise(Data,type):
    
    # Check dimension
    Dim = Data.ndim
    
    # Create an empty matrix
    StdData = np.zeros((Data.shape[0],Data.shape[1]))
    
    # Case of vector input
    if Dim == 1:
        for n in range(0,Data.shape[0]):
            Data = (Data-min(Data))/(max(Data)-min(Data))
            
    # Case of matrix input
    else:
        
    # Standardise by column
        if type == 0:
            for n in range(0,Data.shape[1]):
                DataColumn = Data[:,n]
                DataColumn = (DataColumn-np.mean(DataColumn))/np.std(DataColumn)
                StdData[:,n] = DataColumn
                
    # Standardise by row
        else:
            for m in range(0,Data.shape[0]):
                DataRow = Data[m,:]
                DataRow = (DataRow-np.mean(DataRow))/np.std(DataRow)
                StdData[m,:] = DataRow
                
    return StdData