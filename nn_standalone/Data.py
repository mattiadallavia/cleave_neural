import numpy as np
import pandas as pd
import os

#from sktime.datasets import load_airline
#from sktime.forecasting.model_selection import temporal_train_test_split
#from sktime.performance_metrics.forecasting import smape_loss
#from sktime.utils.plotting import plot_series

class Data():
    def __init__(self, dataset_path, features, N, target_index, feature_index, train_ratio):
            path = dataset_path.split('/')
            dataset_path = os.path.join('data','training_0', 'realisation_0.dat')
            self.n_instances = len(os.listdir(path[0]))
            self.target_index = target_index
            self.feature_index = feature_index
            self.n_features = features
            self.N = N
            self.n_train = int(train_ratio*self.N)

            self.read_data(dataset_path)


    def read_data(self, dataset_path):

        self.data = np.genfromtxt(dataset_path, dtype='float')

        # Insert to train feature vectors
        self.x_train = self.data[:self.n_train, self.feature_index]
        self.x_val = self.data[self.n_train:self.N, self.feature_index]
        self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

        self.y_train = self.data[:self.n_train, self.target_index].reshape(self.n_train,1)
        self.y_val = self.data[self.n_train:self.N, self.target_index].reshape(self.N-self.n_train,1)
        
        



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