#SVM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import process_data as data

def main():
   


    xs_train, ls_train, xs_val, ls_val = data.main("train")
    #xs_val, ls_val = data.main("test")

    xs_train = xs_train.reshape(-1, 784).astype('float32')
    xs_val = xs_val.reshape(-1, 784).astype('float32')
    ls_train = tf.one_hot(ls_train, depth=10)
    ls_val = tf.one_hot(ls_val, depth=10) 

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(28*28,),activation='linear'),
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

    model.compile(optimizer='adam',
              loss='hinge',
              metrics=['accuracy'])
    model.fit(xs_train, ls_train, epochs=125, batch_size=32, validation_data=(xs_val, ls_val))
    

if __name__ == "__main__":
    main()
