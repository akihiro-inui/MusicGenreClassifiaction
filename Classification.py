#==============================================================================
# classification.py
# Program author: Akihiro Inui
# Implement classification algorithm
#==============================================================================
import numpy as np
import pandas as pd
import os
import random
from scipy.spatial import distance
import keras
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import csv
#==============================================================================
# Path to Data
#==============================================================================
path = "[Your Working Directory]"
os.chdir(path)
#==============================================================================
# 1. Load data
#==============================================================================
class Classification:
    def __init__(self,Data):
        Data = pd.read_csv("{0}".format(Data),index_col=0)
        self.Data = Data
#==============================================================================
# 2. Data Order Randomisation
#==============================================================================
    def mkdataset(self,train):
        # Create random number 
        a = list(range(1,self.Data.shape[0]+1))
        randomNum = random.sample(a, len(a))
        d = {'random': randomNum}
        randomNum = pd.DataFrame(data=d)
        
        # Sort by random numbers at the end
        Data = pd.concat([self.Data, randomNum], axis=1)
        Data = Data.sort_values(by="random")
        Data = Data.drop("random", axis = 1)
        
        # Feature dimention
        fdim = Data.shape[1]-1
        
        # Train/Test numbers
        self.TrainNumber = round(Data.shape[0]*train)
        self.TestNumber = Data.shape[0] - self.TrainNumber
        
        # Data/Label Separation
        TrainData = Data.iloc[0:self.TrainNumber,0:fdim].values
        self.TrainData = TrainData
        TrainLabel = Data.iloc[0:self.TrainNumber,fdim].values
        self.TrainLabel = TrainLabel
        TestData = Data.iloc[self.TrainNumber:,0:fdim].values
        self.TestData = TestData
        TestLabel = Data.iloc[self.TrainNumber:,fdim].values
        self.TestLabel = TestLabel
        
        Dataset = [TrainData,TrainLabel,TestData,TestLabel]
        
        return Dataset
    
#==============================================================================
# 3. Fuzzy k-NN
#==============================================================================
    def kNN(self,Dataset,k):
        # Initialisation and Data load
        TrainData = Dataset[0]
        TrainLabel = Dataset[1]
        TestData = Dataset[2]
        TestLabel = Dataset[3]
        TrainNumber = len(TrainLabel)
        TestNumber = len(TestLabel)
        
        classes = int(max(TrainLabel))
        d = np.zeros(TrainNumber)
        beta = 0.5
        U = np.zeros((TrainNumber,classes))
        
        ## Train phase
        
        # Main loop for each Train data
        for n in range(0,TrainNumber):
            # Extract one train vector
            Trainvector = TrainData[n]                                        
        
            # Calculate Euclidean Distance
            for l in range(0,TrainNumber):
                d[l] = distance.euclidean(TrainData[l],Trainvector)            
            Traindistance = np.c_[d,TrainLabel]      
            
            # Sort by distance
            a = Traindistance[Traindistance[:,0].argsort()]  

            # Ignore the nearest one as its always true                                     
            a = np.delete(a,0,0)
            
            # Class in within k
            indx = a[0:k,1]
            
            C = np.zeros(classes)
        
            # Find how many times the class appeared
            for c in range(0,classes):
                C[c] = sum(indx == c+1)
           
            for c in range(0,classes):
                U[n,c] = (1-beta)*C[c]/k
                if c+1 == TrainLabel[n]:
                     U[n,c] = U[n,c] + beta
                     
        ## Test phase     
        # Initialisation
        PreU = np.zeros((TestNumber,classes))
        PreAnswer = np.zeros(TestNumber)

        for m in range(0,TestNumber):
            # Extract one test vector
            Testvector = TestData[m]        
        
            # Calculate Euclidean Distance
            for o in range(0,TrainNumber):
                d[o]  = distance.euclidean(TrainData[o],Testvector) 
                
            ind = np.argpartition(d, k)[:k]          
            Testdistance = np.c_[d,TrainLabel] 
        
            # Sort by distance
            b = Testdistance[Testdistance[:,0].argsort()]  
            
            # Index
            Indx = b[0:k,1]
        
            # Predict answer if the distance is 0
            if Indx[0] == 0:
                PreAnswer[m] = Indx[0]
            else:
                # weights
                w = np.zeros(k)
        
                # Add weights to the nearest vectors
                for kk in range(0,k):
                    # Claculate weights for each distance (Heavier if closer)
                    w[kk] = 1/(d[kk]**2)                         
                    PreU[m,:] =  PreU[m,:] + U[ind[kk],:]*w[kk]  
        
                # Divided by the weights
                PreU[m,:] = PreU[m,:]/sum(w)
        
                # Find the maximum element
                maximel = np.where(PreU[m,:]==max(PreU[m,:]))
                if len(maximel[0]) >=2:
                    maximel = int(maximel[0][0]) + 1
                elif len(maximel[0]) == 0:
                    maximel = 11
                else:
                    maximel = int(maximel[0]) + 1
                PreAnswer[m] = maximel

            # Calculate accuracy
            corrects = np.where(TestLabel[:]==PreAnswer[:])[0]
            accuracy = 100*(len(corrects)/TestNumber)

        return accuracy
#==============================================================================
# 4. Multi Layer Perceptron
#==============================================================================
    def mlp(self,Dataset):
        # Initialisation and Data load
        TrainData = Dataset[0]
        TrainLabel = np.array(Dataset[1],int)
        TestData = Dataset[2]
        TestLabel = np.array(Dataset[3],int)
        
        # Convert label to one hot
        TrainLabel =  np.identity(max(TrainLabel+1))[TrainLabel]
        TrainLabel = TrainLabel[:,1:]
        TestLabel =  np.identity(max(TestLabel+1))[TestLabel]
        TestLabel = TestLabel[:,1:]
        
        # Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=2)
        
        # Set up the model: model
        model = Sequential()
        
        # Learning Rate
        LearningRate = 0.001
        
        # Add the first layer
        model.add(Dense(200, activation='relu', input_shape=(TrainData.shape[1],)))
        
        # Add the second layer
        model.add(Dense(150,activation='relu'))
        
        # Add the second layer
        model.add(Dense(150,activation='relu'))
        
        # Add the output layer
        model.add(Dense(10))
        
        # Create optimizer
        my_optimizer = SGD(lr=LearningRate)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        #model.compile(optimizer=my_optimizer, loss='mean_squared_error',metrics=['accuracy'])
        #model.compile(optimizer=my_optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
        
        
        # Fit the model
        model_training = model.fit(TrainData,TrainLabel,callbacks=[early_stopping_monitor],\
                                   nb_epoch=50,shuffle=False, validation_split=0.1)
        
        # Save model
        #model.save('MGCmlp.h5')
        
        # Load model
        my_model = load_model('MGCmlp.h5') 
        # my_model.summary()
        
        # Calculate predictions: predictions
        #predictions = my_model.predict(TestData)
        
        ## Create the plot for training history
        # Accuracy
        plt.plot(model_training.history['acc'])
        plt.plot(model_training.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        #Loss
        plt.plot(model_training.history['loss'])
        plt.plot(model_training.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        
        # Evaluation with test data
        score = my_model.evaluate(TestData,TestLabel)
        #print('\nTest loss:', score[0])
        #print('Test accuracy:', score[1])
        score = 100*score[1]
        
        return score
#==============================================================================
# 5. Convolutional Neural Network
#==============================================================================    
    def cnn(self,TargetFigFolder):
        # Input target data folder
        Data_Folder = TargetFigFolder
        Train_Folder = Data_Folder + '_Train'
        Test_Folder = Data_Folder + '_Test'
        
        # Read all genre names
        Genres = os.listdir(Train_Folder)
        Num_Classes = len(os.listdir(Train_Folder))
        
        # Number of classes
        if ".DS_Store" in Genres:
            os.remove(Train_Folder + '/.DS_Store')
            Num_Classes = len(os.listdir(Train_Folder)) - 1
        else:
            Num_Classes = len(os.listdir(Train_Folder))
            
        # Size of images
        Img_Size = 128

        # Initialisation
        train_data = []
        train_label = []
        
        # label names
        label_name = []
        
        # Read Train Images from target directory
        for i, d in enumerate(Genres):
            
            # Get file names in a directory
            files = os.listdir(Data_Folder +'_Train/' + d)
            
            # Delete .DS_Store
            if ".DS_Store" in files:
                os.remove(Data_Folder + '_Train/' + d + '/' + '.DS_Store')
                
            # Read data from each image file
            for f in files:
                # Read a image
                #img = cv2.imread(Train_Folder + '/' + d + '/' + f)
                img = Image.open(Train_Folder + '/' + d + '/' + f)
                
                # Convert image to RGB
                img = img.convert("RGB")
                 
                # Convert to numpy array
                img = np.asarray(img)
                
                # Resize it into squared shape
                img = cv2.resize(img, (Img_Size, Img_Size))
                
                # Flatten and normalise it
                img = img.flatten().astype(np.float32)/255.0
                
                # Add the vector to train data
                train_data.append(img)
        
                # Create Label vector
                #tmp = np.zeros(num_classes)
                tmp = 1*i
                train_label.append(tmp)
        
            # Write label name
            label_genre = d
            label_name.append(label_genre)
        
        # make lines and write it as text file
        with open("label.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(label_name))
            
        # transfer them into numpy format
        train_data = np.asarray(train_data)
        train_label = np.asarray(train_label)

        # number for validation data
        val_rate = 0.1
        val_num = int(val_rate*train_label.shape[0])
        
        # fetch data info
        data_num = train_label.shape[0]
        
        # create random order vector
        rand_vec = np.random.permutation(data_num)
        train_index, val_index = rand_vec[val_num:], rand_vec[:val_num]
        
        # Take out data according to random number index
        train_data,val_data = np.ndarray.take(train_data,train_index, axis = 0), \
                              np.take(train_data,val_index, axis = 0)
        train_label,val_label = np.take(train_label,train_index), np.take(train_label,val_index)
        

        # Initialisation
        test_data = []
        test_label = []
         
        # Read Test Images from target directory
        for i, d in enumerate(Genres):
             
             # Get file names in a directory
             files = os.listdir(Data_Folder +'_Test/' + d)
             
             # Delete .DS_Store
             if ".DS_Store" in files:
                 os.remove(Data_Folder + '_Test/' + d + '/.DS_Store')
                 
             # Read data from each image file
             for f in files:
                 # Read a image
                 #img = cv2.imread(Test_Folder + '/' + d + '/' + f)
                 img = Image.open(Test_Folder + '/' + d + '/' + f)
                 
                 # Convert image to RGB
                 img = img.convert("RGB")
                 
                 # Convert to numpy array
                 img = np.asarray(img)
                 
                 # Resize it into squared shape
                 img = cv2.resize(img, (Img_Size, Img_Size))
                 
                 # Flatten and normalise it
                 img = img.flatten().astype(np.float32)/255.0
                 
                 # Add the vector to train data
                 test_data.append(img)
         
                 # Create Label vector
                 #tmp = np.zeros(num_classes)
                 tmp = 1*i
                 test_label.append(tmp)
                 
        test_data = np.asarray(test_data)
        test_label = np.asarray(test_label)

        train_data = train_data.reshape(train_data.shape[0], Img_Size, Img_Size, 3)
        val_data = val_data.reshape(val_data.shape[0], Img_Size, Img_Size, 3)
        test_data = test_data.reshape(test_data.shape[0], Img_Size, Img_Size, 3)
        train_label = keras.utils.np_utils.to_categorical(train_label, Num_Classes)
        val_label = keras.utils.np_utils.to_categorical(val_label, Num_Classes)
        test_label = keras.utils.np_utils.to_categorical(test_label, Num_Classes)

        # Set up the model: model
        model = Sequential()
        
        # Define early_stopping_monitor
        early_stopping_monitor = EarlyStopping(patience=5)
        
        # Input pixel
        input_shape=(Img_Size, Img_Size, 3)
        
        model = Sequential()
        model.add(Conv2D(128, kernel_size=(4, 4),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(256,3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256,3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(Num_Classes, activation='softmax'))
        adam = Adam(lr=1e-4)
        # Compile the model
        model.compile(loss=keras.losses.categorical_crossentropy,\
                      optimizer=adam,metrics=["accuracy"])
        
        # Fit the model
        model_training = model.fit(train_data,train_label,callbacks=[early_stopping_monitor],\
                                   epochs=100,shuffle=False, validation_data=(val_data, val_label))
        
        # Save model
        model.save(path + '/MGCCNN.h5')
        
        # Load model
        my_model = load_model(path + '/MGCCNN.h5') 
        
        # Model summary
        model.summary()
        
        # Calculate predictions: predictions
        predictions = my_model.predict(test_data)
        
        ## Create the plot for training history
        # Accuracy
        plt.plot(model_training.history['acc'])
        plt.plot(model_training.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        #Loss
        plt.plot(model_training.history['loss'])
        plt.plot(model_training.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # Evaluation with test data
        score = my_model.evaluate(test_data, test_label)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        acc = 100*score[1]
        
        return acc
