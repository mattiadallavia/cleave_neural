import time
import os

from Data import Data
from tensorflow import keras

def nn():

    N_real = 2000
    target_index = 9
    feature_index = [2, 3]
    TIME_STEPS = 10

    # Load data
    dataset_path = os.path.join('data', 'training_1')
    data = Data(dataset_path, N_real, target_index, feature_index, TIME_STEPS)

    # Load trained model
    new_model = keras.models.load_model('model4')

    # When we want to evaluate the performance, x = [ang, ang_vel], y = [pid_force]
    # loss = new_model.evaluate(x, y)
    # print('Restored model, loss: {:5.2f}%'.format(100*loss))

    # Predict on test data. x.shape = (None, 200)
    start = time.time()
    y_hat = new_model.predict(x=data.x_test[:1,:])
    evaluated = time.time() - start
    print('Prediction time: ' + str(evaluated) + ' s')

    return y_hat

if __name__ == '__main__':
   y_hat = nn()





