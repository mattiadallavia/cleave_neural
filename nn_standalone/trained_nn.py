from tensorflow import keras

def nn(x):

    # Load trained model
    new_model = keras.models.load_model('model1')

    # When we want to evaluate the performance, x = [ang, ang_vel], y = [pid_force]
    # loss = new_model.evaluate(x, y)
    # print('Restored model, loss: {:5.2f}%'.format(100*loss))

    # Predict on test data. x.shape = (None, 200)
    y_hat = new_model.predict(x=x)

    return y_hat





