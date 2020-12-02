import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import RobustScaler

#from sktime.datasets import load_airline
#from sktime.forecasting.model_selection import temporal_train_test_split
#from sktime.performance_metrics.forecasting import smape_loss
#from sktime.utils.plotting import plot_series

class Data():
    def __init__(self, dataset_path, N, N_real, target_index, feature_index, train_ratio, TIME_STEPS, test=False):

            self.instances = os.listdir(dataset_path)
            self.target_index = target_index
            self.feature_index = feature_index 
            self.N = N      # Number of train observations
            self.n_train = int(train_ratio*N_real)

            if not test:
                self.read_data(dataset_path, N_real)
                self.xTrain = self.x_train
                self.yTrain = self.y_train
                self.xTest = self.x_test
                self.yTest = self.y_test

                #[self.x_train, self.y_train]  = self.time_series_split(self.x_train, self.y_train, time_steps=TIME_STEPS)
                #[self.x_test, self.y_test]  = self.time_series_split(self.x_test, self.y_test, time_steps=TIME_STEPS)
            else:
                [self.x_train, self.x_test] = self.read_bike_data()
                [self.x_train, self.y_train]  = self.time_series_split(self.x_train, self.x_train.cnt, time_steps=TIME_STEPS)
                [self.x_test, self.y_test]  = self.time_series_split(self.x_test, self.x_test.cnt, time_steps=TIME_STEPS)



    def time_series_split(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i: i + time_steps]
            Xs.append(v)
            ys.append(y[i + time_steps])
            #print('X.shape: '+str(X.shape))
            #print('y.shape: '+str(y.shape))
        return np.array(Xs), np.array(ys)


    def read_bike_data(self):
        df = pd.read_csv("london_merged.csv",
                         parse_dates=['timestamp'],
                         index_col="timestamp"
                         )
        df['hour'] = df.index.hour
        df['day_of_month'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        train_size = int(len(df) * 0.9)
        test_size = len(df) - train_size
        train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

        f_columns = ['t1', 't2', 'hum', 'wind_speed']
        f_transformer = RobustScaler()
        f_transformer = f_transformer.fit(train[f_columns].to_numpy())
        train.loc[:, f_columns] = f_transformer.transform(
        train[f_columns].to_numpy()
        )
        test.loc[:, f_columns] = f_transformer.transform(
        test[f_columns].to_numpy()
        )

        cnt_transformer = RobustScaler()
        cnt_transformer = cnt_transformer.fit(train[['cnt']])
        train['cnt'] = cnt_transformer.transform(train[['cnt']])
        test['cnt'] = cnt_transformer.transform(test[['cnt']])

        return train, test


    def read_data(self, dataset_path, N_real):

        first_iteration = True
        first_test_iteration = True
        parse_test_data = False
        n_train , iters = 0, 0

        for i in range(len(self.instances)):
            iters = iters + 1
            #print(iters)
            filename = self.instances[i]
            self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')

            if first_iteration:
                self.x_train = self.data[:self.n_train, self.feature_index]                                         
                self.x_val = self.data[self.n_train:N_real, self.feature_index]
                self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # Target vectors
                self.y_train = self.data[:self.n_train, self.target_index].reshape(self.n_train,1)
                self.y_val = self.data[self.n_train:N_real, self.target_index]
                self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

                first_iteration = False
            
            elif parse_test_data:
                # Test data
                filename = self.instances[i]
                self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')
                if first_test_iteration:
                    self.x_test = self.data[:N_real,self.feature_index]
                    self.y_test = self.data[:N_real,self.target_index]
                    first_test_iteration = False
                else:
                    self.x_test = np.concatenate((self.x_test, self.data[:N_real,self.feature_index]), axis=0)
                    self.y_test = np.concatenate((self.y_test, self.data[:N_real,self.target_index]), axis=0)
                    
                    n_total = n_train + len(self.y_test)

                    # Break when we have N observations
                    if n_total > self.N:
                        break

                

            else:

                # Feature matrices
                self.x_train = np.concatenate((self.x_train,
                                            self.data[:self.n_train, self.feature_index]),
                                            axis=0)                                          
                self.x_val = np.concatenate((self.x_val,
                                            self.data[self.n_train:N_real, self.feature_index]),
                                            axis=0)
                self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # Target vectors
                self.y_train = np.concatenate((self.y_train,
                                            self.data[:self.n_train, self.target_index].reshape(self.n_train,1)),
                                            axis=0)
                y_val = self.data[self.n_train:N_real, self.target_index]
                self.y_val = np.concatenate((self.y_val,
                                            y_val.reshape(y_val.shape[0],1)),
                                            axis=0)
                self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

            
                n_train = len(self.y_train) + len(self.y_val)

                # Break when we have N observations
                if ( (self.N - n_train) < N_real) or (iters == 50):
                #if (self.N - n_train) < 0:
                    parse_test_data = True

        # Test data
        #filename = self.instances[+1]
        #self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')
        #self.x_test = self.data[:N_real,self.feature_index]
        #self.y_test = self.data[:N_real,self.target_index]


    def time_series_split_old(self, dataset_path, N_real):

        first_iteration = True
        first_test_iteration = True
        parse_test_data = False
        n_train = 0

        for i in range(len(self.instances)):
            
            filename = self.instances[i]
            self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')

            if first_iteration:
                self.x_train = self.data[:N_real, self.feature_index]                                         
                #self.x_val = self.data[self.n_train:N_real, self.feature_index]
                #self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # Target vectors
                self.y_train = self.data[:N_real, self.target_index].reshape(self.n_train,1)
                #self.y_val = self.data[self.n_train:N_real, self.target_index]
                #self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

                first_iteration = False
            
            elif parse_test_data:
                # Test data
                filename = self.instances[i]
                self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')
                if first_test_iteration:
                    self.x_test = self.data[:N_real,self.feature_index]
                    self.y_test = self.data[:N_real,self.target_index]
                    first_test_iteration = False
                else:
                    self.x_test = np.concatenate((self.x_test, self.data[:N_real,self.feature_index]), axis=0)
                    self.y_test = np.concatenate((self.y_test, self.data[:N_real,self.target_index]), axis=0)
                    
                    n_total = n_train + len(self.y_test)

                    # Break when we have N observations
                    if n_total > N_real:
                        break

                

            else:

                # Feature matrices
                self.x_train = np.concatenate((self.x_train,
                                            self.data[:N_real, self.feature_index]),
                                            axis=0)                                          
                #self.x_val = np.concatenate((self.x_val,
                #                            self.data[self.n_train:N_real, self.feature_index]),
                #                            axis=0)
                #self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # Target vectors
                self.y_train = np.concatenate((self.y_train,
                                            self.data[:N_real, self.target_index].reshape(N_real,1)),
                                            axis=0)
                #y_val = self.data[self.n_train:N_real, self.target_index]
                #self.y_val = np.concatenate((self.y_val,
                #                            y_val.reshape(y_val.shape[0],1)),
                #                            axis=0)
                #self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

            
                n_train = len(self.y_train) + len(self.y_val)

                # Break when we have N observations
                if (self.N - n_train) < N_real:
                    parse_test_data = True

