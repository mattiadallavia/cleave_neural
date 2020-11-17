import numpy as np
import pandas as pd
import os

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting import plot_series

class Data():
    def __init__(self, dataset_path, sktime_dataset=False):
            self.n_train = 0.7
            self.n_instances = len(os.listdir(dataset_path))
            self.n_features = 2
            self.N = 5000
            
            if sktime_dataset:
                self.read_data_sktime(dataset_path)
            else:
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

    def read_data_sktime(self, dataset_path):
        #self.y = load_airline()
        n_train = int(self.n_train*self.N)

        self.x_train = np.empty((self.n_instances, self.n_features, n_train))
        self.x_val = np.empty((self.n_instances, self.n_features, self.N-n_train))

        self.y_train = np.empty((self.n_instances, 1, n_train))
        self.y_val = np.empty((self.n_instances, 1, self.N-n_train))
        
        " Read dataset and divide into train, validation and test set "
        for i in range(self.n_instances):
            filename = os.listdir(dataset_path)[i]
            self.data = np.genfromtxt(os.path.join(dataset_path,filename), dtype='float', delimiter=',')

            #if  i < (self.n_instances-1):

            # Insert to train feature vectors
            feature_data_train = self.data[:n_train,:self.n_features].transpose()
            feature_data_val = self.data[n_train:self.N,:self.n_features].transpose()

            self.x_train = np.insert(self.x_train, i, feature_data_train, axis=0)
            self.x_val = np.insert(self.x_val, i, feature_data_val, axis=0)

            target_data_train = self.data[:n_train,self.n_features:].transpose()
            target_data_val = self.data[n_train:self.N,self.n_features:].transpose()
            
            self.y_train = np.insert(self.y_train, i, target_data_train, axis=0)
            self.y_val = np.insert(self.y_val, i, target_data_val, axis=0)
                
            #else:
                #pass

        self.x_train = np.delete(self.x_train, [5,6,7,8,9], axis=0)
        self.x_val = np.delete(self.x_val, [5,6,7,8,9], axis=0)

        self.y_train = np.delete(self.y_train, [5,6,7,8,9], axis=0)
        self.y_val = np.delete(self.y_val, [5,6,7,8,9], axis=0)





        """
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
                self.y_test = data.loc[:, ['Force']]   """
