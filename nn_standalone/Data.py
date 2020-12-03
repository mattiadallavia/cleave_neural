import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

class Data():
    def __init__(self, dataset_path, N, N_real, target_index, feature_index, train_ratio, TIME_STEPS):

            self.instances = os.listdir(dataset_path)
            self.target_index = target_index
            self.feature_index = feature_index 
            self.N = N      # Number of train observations
            self.n_train = int(train_ratio*N_real)

            self.read_data(dataset_path, N_real, train_ratio)

            # Save for plot purpose
            self.xTrain = self.x_train
            self.yTrain = self.y_train
            self.xTest = self.x_test
            self.yTest = self.y_test
            self.xVal = self.x_val
            self.yVal = self.y_val

            [self.x_train, self.y_train]  = self.time_series_split(self.x_train, self.y_train, time_steps=TIME_STEPS)
            [self.x_val, self.y_val]  = self.time_series_split(self.x_val, self.y_val, time_steps=TIME_STEPS)
            [self.x_test, self.y_test]  = self.time_series_split(self.x_test, self.y_test, time_steps=TIME_STEPS)
            


    def time_series_split(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i: i + time_steps]
            v = np.concatenate((v[:,0], v[:,1]))
            Xs.append(v)
            ys.append(y[i + time_steps])
            #print('X.shape: '+str(X.shape))
            #print('y.shape: '+str(y.shape))
        return np.array(Xs), np.array(ys)


    def read_data(self, dataset_path, N_real, train_ratio):

        N_real_val = 2000
        #n_train_files = int((self.N/N_real)*train_ratio)
        #n_val_files = math.ceil((self.N/N_real)*(1-train_ratio))
        n_train_files = 47
        n_val_files = 1
        n_test_files = 12
        total_files = n_train_files + n_val_files + n_test_files

        first_train_iteration = True

        for i in range(n_train_files):

            filename = self.instances[i]
            self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')

            # Train data
            if first_train_iteration:
                self.x_train = self.data[:N_real, self.feature_index]

                # Target vectors
                self.y_train = self.data[:N_real, self.target_index].reshape(N_real,1)

                first_train_iteration = False

            else:

                self.x_train = np.concatenate((self.x_train,
                                            self.data[:N_real, self.feature_index]),
                                            axis=0)   

                self.y_train = np.concatenate((self.y_train,
                                            self.data[:N_real, self.target_index].reshape(N_real,1)),
                                            axis=0)

        first_val_iteration = True

        for i in range(n_train_files, (n_train_files + n_val_files)):

            filename = self.instances[i]
            self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')

            # Validation data 
            if first_val_iteration:
                self.x_val = self.data[:N_real_val, self.feature_index]
                self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                self.y_val = self.data[:N_real_val, self.target_index]
                self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

                first_val_iteration = False

            else:
                # self.x_val = self.data[:N_real_val, self.feature_index]
                # self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # self.y_val = self.data[:N_real_val, self.target_index]
                # self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

                self.x_val = np.concatenate((self.x_val,
                                            self.data[:N_real_val, self.feature_index]),
                                            axis=0)
                self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                y_val = self.data[:N_real_val, self.target_index]
                self.y_val = np.concatenate((self.y_val,
                                            y_val.reshape(y_val.shape[0],1)),
                                            axis=0)
                self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))
            

        first_test_iteration = True

        for i in range((n_train_files + n_val_files), total_files):

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

            



    def plot_data(self, fs):

        # Inspect the training data
        fig1 = plt.figure(1)
        fig1.suptitle('Training data')

        ax = fig1.add_subplot(311)   
        ax1 = fig1.add_subplot(312)
        ax2 = fig1.add_subplot(313)

        t_total = (len(self.yTrain)+len(self.yTest))/fs
        t_total = len(self.xTrain)/fs
        t = np.arange(0,t_total,step=0.01)

        # Angle
        ax.plot(t,([x for x in self.xTrain[:,0]]))

        # Angle rate
        ax1.plot(t,([x for x in self.xTrain[:,1]]))

        # Force on cart
        ax2.plot(t,self.yTrain)

        plt.setp(ax, ylabel='Angle [rad]')
        plt.setp(ax1, ylabel='Angular Velocity [rad/s]')
        plt.setp(ax2, ylabel='Force [N]')

        plt.xlabel("Time [s]")
        #plt.show()

        # Inspect the validation data
        fig2 = plt.figure(2)
        fig2.suptitle('Validation data')

        ax = fig2.add_subplot(311)   
        ax1 = fig2.add_subplot(312)
        ax2 = fig2.add_subplot(313)

        t_total = len(self.xVal)/fs
        t = np.arange(0,t_total, step=0.01)

        # Angle
        ax.plot(t,([x for x in self.xVal[:,0]]))

        # Angle rate
        ax1.plot(t,([x for x in self.xVal[:,1]]))

        # Force on cart
        ax2.plot(t,self.yVal)

        plt.setp(ax, ylabel='Angle [rad]')
        plt.setp(ax1, ylabel='Angular Velocity [rad/s]')
        plt.setp(ax2, ylabel='Force [N]')

        plt.xlabel("Time [s]")
        #plt.show()


    def read_data_old(self, dataset_path, N_real, train_ratio):

        n_train_files = int(self.instances*train_ratio)
        n_val_files = int(self.instances*(1-train_ratio))
        n_test_files = 1
        total_files = n_train_files + n_val_files + n_test_files

        first_train_iteration, first_val_iteration = True, True
        first_test_iteration = True
        parse_test_data = False
        n_train , iters = 0, 0

        for i in range(total_files):
            iters = iters + 1
            #print(iters)
            filename = self.instances[i]
            self.data = np.genfromtxt(os.path.join(dataset_path, filename), dtype='float')

            if first_train_iteration:
                self.x_train = self.data[:self.n_train, self.feature_index]                                         
                #self.x_val = self.data[self.n_train:N_real, self.feature_index]
                #self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # Target vectors
                self.y_train = self.data[:self.n_train, self.target_index].reshape(self.n_train,1)
                #self.y_val = self.data[self.n_train:N_real, self.target_index]
                #self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

                first_train_iteration = False

            elif i <= n_train_files:

                self.x_train = np.concatenate((self.x_train,
                                            self.data[:self.n_train, self.feature_index]),
                                            axis=0)   

                self.y_train = np.concatenate((self.y_train,
                                            self.data[:self.n_train, self.target_index].reshape(self.n_train,1)),
                                            axis=0)

            elif n_train_files < i <= (n_train_files + n_val_files):
                self.x_val = self.data[self.n_train:N_real, self.feature_index]
                self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                self.y_val = self.data[self.n_train:N_real, self.target_index]
                self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))
            
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
                # self.x_val = np.concatenate((self.x_val,
                #                             self.data[self.n_train:N_real, self.feature_index]),
                #                             axis=0)
                # self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1]))

                # Target vectors
                self.y_train = np.concatenate((self.y_train,
                                            self.data[:self.n_train, self.target_index].reshape(self.n_train,1)),
                                            axis=0)
                # y_val = self.data[self.n_train:N_real, self.target_index]
                # self.y_val = np.concatenate((self.y_val,
                #                             y_val.reshape(y_val.shape[0],1)),
                #                             axis=0)
                # self.y_val = self.y_val.reshape((self.y_val.shape[0], 1))

            
                # n_train = len(self.y_train) + len(self.y_val)

                # # Break when we have N observations
                # if ( (self.N - n_train) < N_real) or (iters == 50):
                # #if (self.N - n_train) < 0:
                #     parse_test_data = True