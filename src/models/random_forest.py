#Random Forest

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

#If you want to see entire prediction array use this :)
# np.set_printoptions(threshold=sys.maxsize)






def main():
    #getting training data
    train = data.get_data("train")
    y_train = train.iloc[:,0]
    x_train = train.iloc[:,1:]

    randForest = RandomForestClassifier(100)    # (number of estimators, random state)

    randForest.fit(x_train,y_train)

    #getting testing data
    test = data.get_data("test")
    x_test = test.iloc[:,:]

    #testing 
    y_pred = randForest.predict(x_test)
    # print(y_pred)

    # accuracy = randForest.score(x_train, y_train)
    # print('Training accuracy:', accuracy,'----- This is TRAINING ACCURACY NOT TEST ACCURACY!!!!!!!!')

    # print(type(y_pred))
    # print(np.shape(y_pred))
    # print(x_test.shape)


    print('The Random Forest predicts the test data labels to be: ',y_pred)


if __name__ == '__main__':
    print("Hello World!")
    print('Running Random Forest:')
    print()

    main()