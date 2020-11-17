import numpy as np
import pandas as pd
import os

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting import plot_series

class Data():
    def __init__(self, dataset_path, features, N, n_features_temp, train_ratio):
            
            self.n_instances = len(os.listdir(dataset_path))
            self.n_features_temp = n_features_temp
            self.n_features = features
            self.N = N
            self.n_train = int(train_ratio*self.N)

            self.read_data(dataset_path)


    def read_data(self, dataset_path):

        self.x_train = np.empty((self.n_train, self.n_features))
        self.x_val = np.empty((self.N-self.n_train, self.n_features))

        self.y_train = np.empty((self.n_train, 1))
        self.y_val = np.empty((self.N-self.n_train, 1))

        filename = 'manuels_data.csv'
        self.data = np.genfromtxt(os.path.join(dataset_path,filename), dtype='float')

        # Insert to train feature vectors
        feature_data_train = self.data[:self.n_train,2:self.n_features_temp]
        feature_data_val = self.data[self.n_train:self.N,2:self.n_features_temp]

        self.x_train = np.concatenate((self.x_train, feature_data_train), axis=0)
        self.x_val = np.concatenate((self.x_val, feature_data_val), axis=0)

        target_data_train = self.data[:self.n_train,self.n_features_temp:]
        target_data_val = self.data[self.n_train:self.N,self.n_features_temp:]
        
        self.y_train = np.concatenate((self.y_train, target_data_train), axis=0)
        self.y_val = np.concatenate((self.y_val, target_data_val), axis=0)



    def read_data_old(self, dataset_path):
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
            self.data = data
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