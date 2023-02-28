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
    #getting data
    xs_train, ys_train, xs_val, ys_val = data.main("train") 

    randForest = RandomForestClassifier(100)    # (number of estimators)

    randForest.fit(xs_train,ys_train)

    #testing 
    y_pred = randForest.predict(xs_val)
    print('The Random Forest predicts the test data labels to be: ',y_pred)


    accuracy = randForest.score(xs_val, ys_val)
    print('Test accuracy:', accuracy)

if __name__ == '__main__':
    print("Hello World!")
    print('Running Random Forest:')
    print()

    main()