import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from Data import Data
from rbflayer import RBFLayer, InitCentersRandom
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K


def main():

    #print(tf.__version__)
    np.set_printoptions(precision=3, suppress=True)

    # Parameters
    N = 144000                    # Observations for training data (train + test)
    N_real = 2000
    target_index = 9
    feature_index = [2, 3]
    train_ratio = 0.8
    TIME_STEPS = 100
    fs = 100
    #K = 5


    # Hyperparameters
    learning_rate = 0.001
    units = 200
    epochs = 10
    batch_size = 30
    validation_split = 0.2

    # Load data set
    dataset_path = os.path.join('data', 'training_1')
    data = Data(dataset_path, N, N_real, target_index, feature_index, train_ratio, TIME_STEPS)

    data.plot_data(fs)

    # Define model
    model = Sequential()
    #model.add(RBFLayer(units=units, gamma=gamma))

    # This considers previous time observations
    rbflayer = RBFLayer(units,
                        initializer=InitCentersRandom(data.x_train),
                        betas=2.0,
                        #input_shape=(data.x_train.shape[1], data.x_train.shape[2]))
                        input_shape=(data.x_train.shape[1],))


    model.add(rbflayer)

    #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
    #model.add(Dense(1, activation='sigmoid', name='foo', kernel_initializer=initializer))
    #model.add(Dense(1, activation=None, kernel_initializer=initializer))
    #model.add(Dense(units))
    model.add(Dense(1, kernel_regularizer=l2(0.009), use_bias=False))
    #model.add(Dense(1, , kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), use_bias=False))
    
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
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
    plt.plot(np.log10(history.history['loss']), label='Train loss')
    plt.plot(np.log10(history.history['val_loss']), label='Validation loss')
    plt.legend()

    plt.ylabel('log10(loss)')
    plt.xlabel("Epochs")
    #plt.show()


    y_hat = model.predict(x=data.x_test)
    error = y_hat-data.y_test.reshape(data.y_test.shape[0],1)
    MSE = (1/N_real) * np.sum(np.power((y_hat-data.y_test.reshape(data.y_test.shape[0],1)),2))

    # Inspect output
    time = np.arange(len(data.y_test))

    fig4 = plt.figure(4)
    fig4.suptitle('Test data')

    ax3 = fig4.add_subplot(211)   
    ax4 = fig4.add_subplot(212, sharex=ax3)

    ax3.plot(time, data.y_test)
    ax4.plot(time, y_hat)

    plt.setp(ax3, ylabel='True Force [N]')
    plt.setp(ax4, ylabel='Estimated Force [N]')

    plt.xlabel("Time")
    plt.show()

main()




