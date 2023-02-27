#test for my sanity

# KNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import process_data as data





### Actuall KNN

# class KNN:
#     def __init__(self, k):
#         self.k = k

#     def euclidean_dist(self, x1, x2):
#         return np.sqrt(np.sum((x1 - x2)**2))
    
    


def main():
    print("main going!")
    xs_train, ls_train, xs_val, ls_val = data.main("train")
    # xs_train = tf.convert_to_tensor(xs_train, dtype=tf.float32)
    # ys_train = tf.convert_to_tensor(ys_train, dtype=tf.float32)
    # xs_val = tf.convert_to_tensor(xs_val, dtype=tf.float32)
    # ys_val = tf.convert_to_tensor(ys_val, dtype=tf.float32)

    # print(xs_train)



if __name__ == '__main__':
    print("hello world")
    main()