import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from rbflayer import RBFLayer, InitCentersRandom
from Data import Data
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K



class RBFLayer1(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer1, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
         #print(input_shape)
         #print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer1, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def main():

    #print(tf.__version__)
    np.set_printoptions(precision=3, suppress=True)

    # Parameters
    N = 20000                     # Observations for training data (train + test)
    N_real = 5500
    target_index = 9
    feature_index = [2, 3]
    train_ratio = 0.9
    TIME_STEPS = 10
    fs = 100
    K = 5
    TEST = False

    # Hyperparameters
    learning_rate = 0.001
    gamma = 1
    units = 20
    epochs = 100
    batch_size = 30
    validation_split = 0.2

    # Load data set
    dataset_path = os.path.join('data', 'training_1')
    data = Data(dataset_path, N, N_real, target_index, feature_index, train_ratio, TIME_STEPS, test=TEST)


    # Inspect the data
    fig = plt.figure()
    fig.suptitle('Training data')

    ax = fig.add_subplot(211)   
    ax1 = fig.add_subplot(212)
    #ax2 = fig.add_subplot(313)

    t_total = (len(data.yTrain)+len(data.yTest))/fs
    t_total = len(data.xTrain)/fs
    t = np.arange(0,t_total,step=0.01)

    # Angle fold 1
    #ax.plot(t,([x for x in data.x_train[0][:,0]] + [None for x in data.x_train[1][:,0]] + [None for x in data.x_train[2][:,0]]))
    #ax.plot(t,([None for x in data.x_train[0][:,0]] + [x for x in data.x_train[1][:,0]] + [None for x in data.x_train[2][:,0]]))
    #ax.plot(t,([None for x in data.x_train[0][:,0]] + [None for x in data.x_train[1][:,0]] + [x for x in data.x_train[2][:,0]]))
    ax.plot(t,([x for x in data.xTrain[:,0]]))


    #ax.plot(t,([x for x in data.x_train[:][:,0]] + [None for i in data.x_test[:,0]]))
    #ax.plot(t,([None for i in data.x_train[:,0]] + [x for x in data.x_test[:,0]]))
    #ax.set_xlim([0, t_total])

    # Angle fold 2
    #ax.plot(t,([x for x in data.x_train[:,0]] + [None for i in data.x_test[:,0]]))
    #ax.plot(t,([None for i in data.x_train[:,0]] + [x for x in data.x_test[:,0]]))
    #ax.set_xlim([0, t_total])

    # Angle fold 3
    #ax.plot(t,([x for x in data.x_train[:,0]] + [None for i in data.x_test[:,0]]))
    #ax.plot(t,([None for i in data.x_train[:,0]] + [x for x in data.x_test[:,0]]))
    #ax.set_xlim([0, t_total])

    

    # Angle rate
    #ax1.plot(t,([x for x in data.x_train[0][:,1]] + [None for x in data.x_train[1][:,1]] + [None for x in data.x_train[2][:,1]]))
    #ax1.plot(t,([None for x in data.x_train[0][:,1]] + [x for x in data.x_train[1][:,1]] + [None for x in data.x_train[2][:,1]]))
    #ax1.plot(t,([None for x in data.x_train[0][:,1]] + [None for x in data.x_train[1][:,1]] + [x for x in data.x_train[2][:,1]]))
    ax1.plot(t,([x for x in data.xTrain[:,1]]))

    # Force on cart
    #ax2.plot(t, ([x for x in data.y_train] + [None for i in data.y_test]))
    #ax2.plot(t, ([None for i in data.y_train] + [y for y in data.y_test]))


    plt.setp(ax, ylabel='Angle')
    plt.setp(ax1, ylabel='Angular Velocity')
    #plt.setp(ax2, ylabel='Force')

    plt.xlabel("Time")
    #plt.show()


    # Define model
    model = Sequential()
    #model.add(RBFLayer(units=units, gamma=gamma))
    #X = np.array(([x for x in data.x_train[:,0]]+[x for x in data.x_train[:,1]]))

    # This works
    #rbflayer = RBFLayer(10,
    #                    initializer=InitCentersRandom(data.xTrain),
    #                    betas=2.0,
    #                    #input_shape=(data.x_train.shape[1], data.x_train.shape[2]))
    #                    input_shape=(data.xTrain.shape[1],))

    # This considers previous time observations
    rbflayer = RBFLayer(units,
                        initializer=InitCentersRandom(data.x_train),
                        betas=2.0,
                        #input_shape=(data.x_train.shape[1], data.x_train.shape[2]))
                        input_shape=(data.x_train.shape[1],))


    model.add(rbflayer)

    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
    #model.add(Dense(1, activation='sigmoid', name='foo', kernel_initializer=initializer))
    #model.add(Dense(1, activation=None, kernel_initializer=initializer))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()

    #history = model.fit(x=data.x_train, 
    #                    y=data.y_train,
    #                    epochs=epochs,
    #                    batch_size=batch_size,
    #                    validation_split=validation_split,
    #                    #validation_data=(data.x_val, data.y_val),
    #                    shuffle=False)

    history = model.fit(x=data.x_train, 
                        y=data.y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        #validation_data=(data.x_val, data.y_val),
                        shuffle=False)

    fig2 = plt.figure(2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    #plt.show()


    y_hat = model.predict(x=data.x_test)
    error = y_hat-data.y_test.reshape(data.y_test.shape[0],1)
    MSE = (1/N_real) * np.sum(np.power((y_hat-data.y_test.reshape(data.y_test.shape[0],1)),2))
    #print(data.y_test)
    #print(y_hat)

    #print('MSE: '+str(MSE))
    #print('Sandard deviation true force: '+str(np.std(data.y_test.reshape(data.y_test.shape[0],1))))

    # Inspect output
    time = np.arange(len(data.y_test))
    fig3 = plt.figure(3)
    fig3.suptitle('Test data')

    ax2 = fig3.add_subplot(211)   
    ax3 = fig3.add_subplot(212)

    ax2.plot(time, data.y_test)
    ax3.plot(time, y_hat)

    plt.setp(ax2, ylabel='True Force')
    plt.setp(ax3, ylabel='Estimated Force')

    plt.xlabel("Time")
    plt.show()




main()




