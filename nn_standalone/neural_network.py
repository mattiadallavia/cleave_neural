import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sktime

from warnings import simplefilter
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting import plot_series

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from Data import Data
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer, Flatten, Dense, Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import RMSprop
#from rbf_keras.rbflayer import RBFLayer, InitCentersRandom
#%matplotlib inline
simplefilter("ignore", FutureWarning)

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        #print(input_shape)
        #print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * 12)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)



def main():

    print(tf.__version__)
    np.set_printoptions(precision=3, suppress=True)

    # Hyperparameters
    learning_rate = 0.1
    gamma = 0.5
    units = 10
    epochs = 300
    batch_size = 10

    # Load data set
    dataset = 'train_dataset'
    #data = Data(dataset)
    data = Data(dataset, sktime_dataset=True)
    #plt.figure()
    n_time = 200
    n_train = int(data.n_train*data.N)
    time = np.arange(n_time)
    instance_features = [time, data.x_train[0,0,:n_time], time, data.x_train[1,0,:n_time]]

    #plot_series(pd.Series(one_feature).reset_index(drop=True))

    plt.plot(time, data.x_train[0,0,:n_time], label='Instance one')
    plt.plot(time, data.x_train[1,0,:n_time], label='Instance two')
    plt.plot(time, data.x_train[2,0,:n_time], label='Instance three')
    plt.plot(time, data.x_train[3,0,:n_time], label='Instance four')
    plt.plot(time, data.x_train[4,0,:n_time], label='Instance five')
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.legend()
    plt.show()

    # Inspect the data. 
    # TODO: Use this for the presentation when we have better data
    plot_cols = ['Angle', 'Angular Velocity', 'Force']
    plot_features = data.data[plot_cols][3000:5000]

    plot_features.plot(subplots=True)
    #plt.show()

    sns.pairplot(data.x_train[['Angle', 'Angular Velocity']], diag_kind='kde')
    #plt.show()

    # Look at overall statistics
    #print(data.x_train.describe().transpose())

    """
    # TODO: Should we normalize the data? This doesn't seem to work.
    # Normalize features that use different scale and ranges 
    # Makes training more stable
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(data.x_train))

    # When the layer is called it returns the input data
    # with each feature independently normalized
    print(normalizer.mean.numpy())

    first = np.array(data.x_train[:1])
    with np.printoptions(precision=2, suppress=True):
        #print('First example:', first)
        #print()
        #print('Normalized:', normalizer(first).numpy())
    """

    # Define the model
    model = Sequential()
    model.add(Input(shape=((2,))))
    model.add(Dense(120, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(120,  kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))


    #model.add(Input(shape=(2,)))
    #model.add(RBFLayer(units=units, gamma=gamma))
    #model.add(Dense(1))
    #print(model.summary())


    # Configure training procedure 
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    #model.compile(optimizer='rmsprop', loss=binary_crossentropy)

    # Execute the training
    #%%time
    history = model.fit(
        data.x_train[['Angle', 'Angular Velocity']], 
        data.y_train['Force'],
        epochs=epochs,
        batch_size=batch_size,
        # Logging
        verbose=1,
        validation_data=(data.x_val, data.y_val)
    )


main()




