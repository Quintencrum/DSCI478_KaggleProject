import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import process_data as data


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def main(epochs: int=20, verbose=0):
    xs_train, ls_train, xs_val, ls_val = data.main("train")
    
    # Preprocess the data
    xs_train = xs_train.reshape(-1, 28*28).astype('float32')
    ls_train = tf.one_hot(ls_train, depth=10)
    xs_val = xs_val.reshape(-1, 28*28).astype('float32')
    ls_val = tf.one_hot(ls_val, depth=10)

    # Define the logistic regression model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(28*28,), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    ## train_history.history['accuracy'] and train_history.history['loss']
    ## train_history.history['val_accuracy'] and train_history.history['val_loss']
    train_history = model.fit(xs_train, ls_train, epochs=epochs, batch_size=32, validation_data=(xs_val, ls_val),
                              shuffle=True, verbose=verbose)

    # Evaluate the model on the validation set
    if verbose != 0:
        loss, accuracy = model.evaluate(xs_val, ls_val)
        print(f'\nValidation loss: {loss}, Validation accuracy: {accuracy}')

    return model, train_history

    

