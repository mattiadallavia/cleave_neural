import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

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
    learning_rate = 0.5
    gamma = 0.1
    units = 10
    epochs = 3
    batch_size = 256

    # Load data set
    dataset = 'train_dataset'
    data = Data(dataset)

    # Inspect the data
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
    #model.add(tf.keras.Input(shape=((2,))))
    #model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
    #model.add(Input(shape=(2,)))
    model.add(Dense(20, input_shape=(2,)))
    model.add(RBFLayer(units=units, gamma=gamma))
    print(model.summary())


    # Run the untrained model
    #print(model.predict(data.x_train.loc[:9]))

    # Configure training procedure 
    """model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )"""

    model.compile(optimizer='rmsprop', loss=binary_crossentropy)

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




