import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from Data import Data
from rbflayer import RBFLayer, InitCentersRandom
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

def cross_validation():
    
    np.set_printoptions(precision=3, suppress=True)

    # Parameters
    N = 144000                    # Observations for training data (train + test)
    N_real = 2000
    target_index = 9
    feature_index = [2, 3]
    train_ratio = 0.8
    fs = 100
    dataset_path = os.path.join('data', 'training_1')


    # Hyperparameters
    learning_rate = 0.001

    units = 100
    TIME_STEPS = 100        # Optimal
    sigma2_Kmeans = 0.003
    # sigma2_Kmeans = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]     # Find optimal
    # sigma2_Kmeans = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.003, 0.005, 0.007, 0.009, 0.02, 0.04, 0.06, 0.08, 0.1, 0.3, 0.5, 0.7, 0.9]     # Find optimal
    # sigma2_Kmeans = [0.001, 0.005, 0.01]     # Find optimal
    # reg_lambda = 0.01
    reg_lambda = [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.003, 0.005, 0.007, 0.009, 0.02, 0.04, 0.06, 0.08, 0.1, 0.3, 0.5, 0.7, 0.9]     # Find optimal
    epochs = 10
    batch_size = 30
    validation_split = 0.2

    # Lists to store MSE
    MSE_list_train = []
    MSE_list_val = []

    # Load data set
    data = Data(dataset_path, N, N_real, target_index, feature_index, train_ratio, TIME_STEPS)


    for i in range(len(sigma2_Kmeans)):

        # Define model
        model = Sequential()

        # This considers previous time observations
        rbflayer = RBFLayer(units,
                            initializer=InitCentersRandom(data.x_train),
                            betas=1/sigma2_Kmeans[i],
                            input_shape=(data.x_train.shape[1],))


        model.add(rbflayer)
        model.add(Dense(1, kernel_regularizer=l2(reg_lambda)))
        model.add(Dense(1, kernel_regularizer=l2(reg_lambda), use_bias=False))
        
        model.compile(loss='mean_squared_error', optimizer=RMSprop())
        model.summary()
        print('Iteration: '+str(i))
        print('sigma2 = '+str(sigma2_Kmeans[i]))

        history = model.fit(x=data.x_train, 
                            y=data.y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            #validation_data=(data.x_val, data.y_val),
                            shuffle=False)

        # Save MSE
        MSE_train = np.sum(history.history['loss'][1:])/epochs
        MSE_val = np.sum(history.history['val_loss'][1:])/epochs
        print('Log(MSE_train): '+str(np.log10(MSE_train)))
        print('Log(MSE_val): '+str(np.log10(MSE_val)))
        MSE_list_train.append(MSE_train)
        MSE_list_val.append(MSE_val)

    # Plot MSE over sigma2
    fig3 = plt.figure(1)
    fig3.suptitle('MSE for different gaussian variances')
    plt.plot(np.log10(MSE_list_train), label='Train loss')
    plt.plot(np.log10(MSE_list_val), label='Validation loss')
    plt.legend()
    
    plt.ylabel('Log(MSE)')
    plt.xlabel(r"$\sigma^2$")
    plt.show()



def main():

    run_cross_validation = False

    if run_cross_validation:
        cross_validation()

    else:
        #print(tf.__version__)
        np.set_printoptions(precision=3, suppress=True)

        # Parameters
        N_real = 2000
        target_index = 9
        feature_index = [2, 3]
        fs = 100
        #K = 5
        save = True

        # Hyperparameters
        learning_rate = 0.001

        units = 100
        TIME_STEPS = 10        # Optimal
        sigma2_Kmeans = 0.003     # Find optimal
        sigma2_weights = 0.01
        reg_lambda = 0.01
        epochs = 40
        batch_size = 30
        validation_split = 0.2

        # Load data set
        dataset_path = os.path.join('data', 'training_1')
        save_path = 'model3'
        data = Data(dataset_path, N_real, target_index, feature_index, TIME_STEPS)

        data.plot_data(fs)

        # Define model
        model = Sequential()

        # This considers previous time observations
        rbflayer = RBFLayer(units,
                            initializer=InitCentersRandom(data.x_train),
                            betas=1/sigma2_Kmeans,
                            input_shape=(data.x_train.shape[1],))


        model.add(rbflayer)

        #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=sigma2_weights)
        model.add(Dense(1))
        model.add(Dense(1, use_bias=False))
        #model.add(Dense(1, , kernel_regularizer=l2(reg_lambda), bias_regularizer=l2(reg_lambda), use_bias=False))
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=RMSprop())
        model.summary()

        history = model.fit(x=data.x_train, 
                            y=data.y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            #validation_data=(data.x_val, data.y_val),
                            shuffle=False)

        fig3 = plt.figure(3)
        fig3.suptitle('Train and validation loss')
        # plt.plot(history.history['loss'], label='Train loss')
        # plt.plot(history.history['val_loss'], label='Validation loss')
        MSE = np.sum(history.history['loss'])
        MSE = MSE/epochs
        print('Log(MSE): '+str(np.log10(MSE)))
        plt.plot(np.log10(history.history['loss']), label='Train loss')
        plt.plot(np.log10(history.history['val_loss']), label='Validation loss')
        plt.legend()
        
        plt.ylabel('Log(loss)')
        #plt.ylabel('Loss')
        plt.xlabel("Epochs")
        #plt.show()

        start = time.time()
        y_hat = model.predict(x=data.x_test)
        elapsed = time.time() - start
        print('Prediction time: '+str(elapsed))
        y_hat = y_hat.reshape(y_hat.shape[0], 1)
        y_test = data.y_test.reshape(data.y_test.shape[0], 1)

        #error = y_hat-data.y_test.reshape(data.y_test.shape[0],1)
        #MSE = (1/N_real) * np.sum(np.power((y_hat-data.y_test.reshape(data.y_test.shape[0],1)),2))

        # Inspect test output
        times = np.arange(len(data.y_test))


        # Figure 4 - y(t)
        fig4 = plt.figure(4)
        fig4.suptitle('Test data')

        ax3 = fig4.add_subplot(211)   
        ax4 = fig4.add_subplot(212, sharex=ax3)

        ax3.plot(times, data.y_test)
        ax4.plot(times, y_hat)

        plt.setp(ax3, ylabel='True Force [N]')
        plt.setp(ax4, ylabel='Estimated Force [N]', ylim=(-0.01,0.11))

        plt.xlabel("Time")

        # Figure 5 - y(angle)
        fig5 = plt.figure(5)
        fig5.suptitle('Test data')

        ax5 = fig5.add_subplot(211)   
        ax6 = fig5.add_subplot(212, sharex=ax5)

        ax5.plot(data.xTest[:23990,0], y_test)
        ax6.plot(data.xTest[:23990,0], y_hat)

        plt.setp(ax5, ylabel='True Force [N]')
        plt.setp(ax6, ylabel='Estimated Force [N]', ylim=(0,0.1))

        plt.xlabel("Angle")

        # Figure 6 - y(angular_vel)
        fig6 = plt.figure(6)
        fig6.suptitle('Test data')

        ax7 = fig6.add_subplot(211)   
        ax8 = fig6.add_subplot(212, sharex=ax7)

        ax7.plot(data.xTest[:23990,1], y_test)
        ax8.plot(data.xTest[:23990,1], y_hat)

        plt.setp(ax7, ylabel='True Force [N]')
        plt.setp(ax8, ylabel='Estimated Force [N]', ylim=(0,0.1))

        plt.xlabel("Anglular Velocity")

        # Figure 7 - center points and widths
        plt.figure(7)
        centers = rbflayer.get_weights()[0]
        widths = rbflayer.get_weights()[1]
        x = centers[:,0]
        y = np.zeros(len(centers))
        plt.scatter(x, y, s=20*widths)

        plt.show()

        if save:
            model.save(save_path)

main()




