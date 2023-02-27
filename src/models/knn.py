# KNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    
    


def main():
    train = data.get_data("train")
    # print(train)
    # print(train.dtypes)
    print(train.iloc[0,0])  #[image num, label value or pixel value]



if __name__ == '__main__':
    print("hello world")
    main()