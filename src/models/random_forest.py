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

from subprocess import call
from sklearn.tree import plot_tree


#If you want to see entire prediction array use this :)
# np.set_printoptions(threshold=sys.maxsize)

def rf():
    #getting data
    xs_train, ys_train, xs_val, ys_val = data.main("train") 

    randForest = RandomForestClassifier(100)    # (number of estimators)

    randForest.fit(xs_train,ys_train)

    #testing 
    y_pred = randForest.predict(xs_val)
    print('The Random Forest predicts the test data labels to be: ',y_pred)


    accuracy = randForest.score(xs_val, ys_val)
    print('Test accuracy:', accuracy)
    return randForest


def visualizationOfRF(model):
    estimator = model.estimators_[5]
    # print(type(estimator))
    classNames = list(np.array([0,1,2,3,4,5,6,7,8,9], dtype='<U4'))
    featureNames = list(np.asarray([f'pixel{i}' for i in range(28*28)]))

    fig = plt.figure(figsize=(15, 10))

    plot_tree(estimator, 
        feature_names=featureNames,
        class_names = classNames,
        filled=True, impurity=True, 
        rounded=True)

    fig.savefig('randomForest.png')





def main():
    randForest = rf()
    visualizationOfRF(randForest)

if __name__ == '__main__':
    print("Hello World!")
    print('Running Random Forest:')
    print()

    main()