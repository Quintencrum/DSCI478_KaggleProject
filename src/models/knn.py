# KNN
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

from sklearn.metrics import accuracy_score

# setting path to import process data
path2 = os.getcwd() + "/src"
sys.path.append(path2)

import process_data as data




### Actual KNN
class KNN:
    def __init__(self, k):
        self.k = k

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def fit(self, x_train, y_train):
        # self.x_train = x_train.values
        # self.y_train = y_train.values
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        # x_test = x_test.values
        x_test = x_test
        y_pred = [self.check_dist_knn(x) for x in x_test]
        return np.array(y_pred)

    def check_dist_knn(self, x):
        distances = [self.euclidean_dist(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common_digit = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_digit
    
    


def main():
    #getting training data
    xs_train, ys_train, xs_val, ys_val = data.main("train",0.01)

    #creating KNN object and training
    knn = KNN(k=3)
    knn.fit(xs_train, ys_train)  
    print('here 1')


    #testing 
    y_pred = knn.predict(xs_val)
    print(y_pred)
    print('The KNN predicts the test data labels to be: ',y_pred)

    #Checking accuracy
    accuracy = accuracy_score(ys_val, y_pred)
    print("Accuracy:", accuracy)
    print('here 3')


if __name__ == '__main__':
    print("Hello World!")
    print('Running KNN:')
    print()
    main()