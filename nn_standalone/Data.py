import numpy as np
import pandas as pd
import os

class Data():
    def __init__(self, dataset_path):
        self.n_train = 0.7
        self.N = 5000 # Number of observations per simulation file (in case they are not equal)
        self.read_data(dataset_path)

    def read_data(self, dataset_path):
        " Read dataset and divide into train, validation and test set "
        
        # Initialize data frames
        self.x_train = pd.DataFrame(columns=['Angle','Angular Velocity'])
        self.x_val = pd.DataFrame(columns=['Angle','Angular Velocity'])

        self.y_train = pd.DataFrame(columns=['Force'])
        self.y_val = pd.DataFrame(columns=['Force'])

        n_files = len(os.listdir(dataset_path))

        for i in range(len(os.listdir(dataset_path))):
            filename = os.listdir(dataset_path)[i]
            data = pd.read_csv(os.path.join(dataset_path,filename), names=['Angle','Angular Velocity','Force']).astype('float64')  
            n_train = int(self.n_train*self.N)

            # Extract train and val data
            if  i < (n_files-1):
                self.x_train = pd.concat([self.x_train, data.loc[0:n_train, ['Angle','Angular Velocity']]]).reset_index(drop=True).astype('float64')
                self.x_val = pd.concat([self.x_val, data.loc[n_train:self.N,  ['Angle','Angular Velocity']]]).reset_index(drop=True).astype('float64')

                self.y_train = pd.concat([self.y_train, data.loc[0:n_train,  ['Force']]]).reset_index(drop=True).astype('float64')
                self.y_val = pd.concat([self.y_val, data.loc[n_train:self.N, ['Force']]]).reset_index(drop=True).astype('float64')
            # Extract test data from last file
            else:
                self.x_test = data.loc[:, ['Angle','Angular Velocity']]
                self.y_test = data.loc[:, ['Force']]                
