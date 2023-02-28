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

from sklearn.tree import export_graphviz
from subprocess import call


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
    print(type(estimator))
    classNames = np.array([0,1,2,3,4,5,6,7,8,9]).toList()
    featureNames = np.asarray([f'pixel{i}' for i in range(28*28)]).toList()
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot', 
                    feature_names = featureNames,
                    class_names = classNames,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    # # Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename = 'tree.png')





def main():
    randForest = rf()
    visualizationOfRF(randForest)

if __name__ == '__main__':
    print("Hello World!")
    print('Running Random Forest:')
    print()

    main()