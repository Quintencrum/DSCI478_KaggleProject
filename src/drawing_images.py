#Drawing image

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

# # setting path to import process data
# path2 = os.getcwd() + "/src"
# sys.path.append(path2)

import process_data as data

def draw(n):
    plt.imshow(n,cmap=plt.cm.binary)
    plt.show()
    print('plt.show')

def draw_one_to_nine():
    xs_train, ys_train, xs_val, ys_val = data.main("train")

    inds = np.array([0,1,2,3,4,5,6,7,8,9])

    for index,element in enumerate(inds):
        for indexj,elementj in enumerate(ys_train):
            if element == elementj:
                inds[index] = indexj
                break
    
    print('are you still alive python')

    # print(inds[0])
    
    # print(xs_train[inds[0]].reshape(28,28))
    # draw(xs_train[inds[0]].reshape(28,28))

    for i in inds:
        draw(xs_train[i].reshape(28,28))


     


def main():
    #getting training data
    xs_train, ys_train, xs_val, ys_val = data.main("train") 

    # draw(xs_train[0].reshape(28,28))
    draw_one_to_nine()



if __name__ == '__main__':
    print("Hello World!")
    print('Printing some images yo:')
    print()

    main()