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

def draw(n,imageName):
    plt.imshow(n,cmap=plt.cm.binary)
    plt.savefig(imageName)
    

def draw_one_to_nine():
    xs_train, ys_train, xs_val, ys_val = data.main("train")

    inds = np.array([0,1,2,3,4,5,6,7,8,9])

    for index,element in enumerate(inds):
        for indexj,elementj in enumerate(ys_train):
            if element == elementj:
                inds[index] = indexj
                break


    draw(xs_train[inds[0]].reshape(28,28),'image_zero.png')
    draw(xs_train[inds[1]].reshape(28,28),'image_one.png')
    draw(xs_train[inds[2]].reshape(28,28),'image_two.png')
    draw(xs_train[inds[3]].reshape(28,28),'image_three.png')
    draw(xs_train[inds[4]].reshape(28,28),'image_four.png')
    draw(xs_train[inds[5]].reshape(28,28),'image_five.png')
    draw(xs_train[inds[6]].reshape(28,28),'image_six.png')
    draw(xs_train[inds[7]].reshape(28,28),'image_seven.png')
    draw(xs_train[inds[8]].reshape(28,28),'image_eight.png')
    draw(xs_train[inds[9]].reshape(28,28),'image_nine.png')
    
    print('done')


def main(drawOneToNine):
    
    if drawOneToNine:
        draw_one_to_nine()
    else:
        print('Nothing will happen :)')



if __name__ == '__main__':
    print("Hello World!")
    print('Printing some images yo:')
    print()

    main()