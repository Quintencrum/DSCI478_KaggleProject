# Q's "Neural Net"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

import sys
import os

# setting path to import process data
path2 = os.getcwd() + "/src"
sys.path.append(path2)

import process_data as data

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier


def nn():
    #getting training data
    xs_train, ys_train, xs_val, ys_val = data.main("train") 

    # model
    model = tf.keras.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(xs_train, ys_train, epochs=20,shuffle=True)

    #testing 
    y_pred = model.predict(xs_val)
    print('Q\'s Neural Network predicts the test data labels to be: ',y_pred)

    #accuracy
    loss, accuracy = model.evaluate(xs_val, ys_val)
    print('Test accuracy:', round(accuracy,2))









def main():
    nn()



if __name__ == '__main__':
    print("Hello World!")
    print('Running Q\'s \"Neural Net\":')
    print()

    main()