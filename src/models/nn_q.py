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







def main():
    #getting training data
    train = data.get_data("train")
    y_train = train.iloc[:,0]
    x_train = train.iloc[:,1:]

    

    #getting testing data
    test = data.get_data("test")
    x_test = test.iloc[:,:]

    # model
    model = tf.keras.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(x_train, y_train, epochs=10)

    #getting testing data
    test = data.get_data("test")
    x_test = test.iloc[:,:]


    #testing 
    y_pred = model.predict(x_test)
    print(y_pred)


    # print('The Random Forest predicts the test data labels to be: ',y_pred)


if __name__ == '__main__':
    print("Hello World!")
    print('Running Q\'s \"Neural Net\":')
    print()

    main()