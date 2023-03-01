import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import process_data as data


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def main(epochs=20, verbose=0):
    xs_train, ls_train, xs_val, ls_val = data.main("train")

    # Preprocess the data
    xs_train = xs_train.reshape(-1, 28, 28).astype('float32')
    #ls_train = tf.one_hot(ls_train, depth=10)
    xs_val = xs_val.reshape(-1, 28, 28).astype('float32')
    #ls_val = tf.one_hot(ls_val, depth=10)

    # Define some layers
    conv_layer = layers.Conv1D(32, (3), activation='relu', input_shape=(28, 28),
                               use_bias=True, padding="same", name="InputLayer")
    pool_layer = layers.MaxPooling1D(pool_size=(2), strides=(1),
                                     padding="valid", name="PoolLayer")

    # Build the NN model
    NN = tf.keras.Sequential([conv_layer, pool_layer,
                              layers.Conv1DTranspose(64, (3), activation='relu',
                                                     name="ConvLayer"),
                              layers.Flatten(),
                              layers.Dense(26, activation='tanh'),
                              layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    NN.compile(optimizer='adam', metrics=['accuracy'],
               loss=tf.keras.losses.SparseCategoricalCrossentropy())

    # Train the model
    train_history = NN.fit(xs_train, ls_train, validation_data=(xs_val, ls_val),
                              epochs=epochs, batch_size=32, shuffle=True, verbose=verbose)

    # Evaluate the model on the validation set
    if verbose != 0:
        loss, accuracy = NN.evaluate(xs_val, ls_val)
        print(f'\nValidation loss: {loss}, Validation accuracy: {accuracy}')       

    return NN, train_history


