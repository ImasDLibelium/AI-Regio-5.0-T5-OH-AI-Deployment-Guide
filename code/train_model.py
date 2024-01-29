#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from scipy import stats

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

print(tf.__version__)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()
    plt.show()


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


def get_model(n_neurons=2, learning_rate=0.001, alpha=0.3):
    model = keras.Sequential([
        layers.Dense(n_neurons, activation='relu', input_shape=[6]),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def workflow(short=False):
    dataset = pd.read_csv(
        './SanBasilio_Lite.csv')
    dataset = dataset.dropna()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset[f'O3_t1']
    test_labels = test_dataset[f'O3_t1']

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    normed_train_data = normed_train_data.loc[:, ['PM10', 'PM2.5', 'NO2', 'SO2', 'O3', 'CO']]
    normed_test_data = normed_test_data.loc[:, ['PM10', 'PM2.5', 'NO2', 'SO2', 'O3', 'CO']]

    param_distribs = {
        "alpha": [0.1, 0.2, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }

    EPOCHS = 1000
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(get_model)

    if short:
        model = get_model()
    else:
        model = RandomizedSearchCV(keras_reg, param_distribs, n_iter=1, verbose=5, cv=3)

    history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])

    plot_history(history)

    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    print(stats.pearsonr(test_labels, test_predictions))
    print(r2_score(test_labels, test_predictions))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('./O3_predictor.tflite',
              "wb") as f:
        f.write(tflite_model)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


# In[3]:

if __name__ == "__main__":
    workflow(short=True)
