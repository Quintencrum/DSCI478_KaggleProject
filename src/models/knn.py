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

# setting path to import process data
path2 = os.getcwd() + "/src"
sys.path.append(path2)

import process_data as data




### Actuall KNN

class KNN:
    def __init__(self, k):
        self.k = k

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def fit(self, x_train, y_train):
        self.x_train = x_train.values
        self.y_train = y_train.values

    def predict(self, x_test):
        x_test = x_test.values
        y_pred = [self.check_dist_knn(x) for x in x_test]
        return np.array(y_pred)

    def check_dist_knn(self, x):
        distances = [self.euclidean_dist(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common_digit = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_digit
    
    


def main():
    train = data.get_data("train")
    # train_nump = train.to_numpy()
    # print(train)
    # print(train.dtypes)
    # print(train.iloc[0,0])  #[image num, label value or pixel value]
    # print(train.iloc[0])
    # print(type(train))

    # print(train_nump[0])
    # print(train.iloc[:,0])

    y_train = train.iloc[:,0]
    x_train = train.iloc[:,1:]

    # data = pd.DataFrame({'X1': [1, 4, 7, 10], 'X2': [2, 5, 8, 11], 'X3': [3, 6, 9, 12], 'y': [0, 1, 1, 0]})
    # X_train = data.iloc[:, :-1]
    # y_train = data.iloc[:, -1]
    # print(y_train)

    knn = KNN(k=3)
    knn.fit(x_train, y_train)


    test = data.get_data("test")
    x_test = test.iloc[:,:]


    y_pred = knn.predict(x_test)

    print(y_pred)



if __name__ == '__main__':
    print("hello world")
    print('')
    main()