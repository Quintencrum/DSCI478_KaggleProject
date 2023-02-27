# KNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def get_data_path():
    print(Path.cwd().joinpath("data"))
    return Path.cwd().joinpath("data")

get_data_path()

# RAW_DATA_FILES = {"test": "test.csv", "train": "train.csv"}
PROCESSED_DATA_FILES = {"xs_t": "xs_train.csv", "labels_t": "labels_train.csv",
                        "xs_v": "xs_validate.csv", "labels_v": "labels_validate.csv"}

# import process_data as data
# def get_data(ftype: str):
#     raw_df = pd.read_csv(utils.get_data_path().joinpath("raw",RAW_DATA_FILES[ftype]),
#                      sep=',', header=[0], dtype=int)
#     assert not np.any(raw_df.isna()), f"Nulls in the dataset {datafiles[ftype]}"
    
#     return raw_df

def get_training_data():
    process_dir = get_data_path().joinpath("processed")
    
    xs_train = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["xs_t"]),
                           sep=',', header=[0], dtype='float32').values
    labels_train = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["labels_t"]),
                               sep=',', header=[0], dtype='float32').values.reshape((-1,))
    xs_validation = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["xs_v"]),
                                sep=',', header=[0], dtype='float32').values
    labels_validation = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["labels_v"]),
                                    sep=',', header=[0], dtype='float32').values.reshape((-1,))

    return xs_train, labels_train, xs_validation, labels_validation







### Actuall KNN

class KNN:
    def __init__(self, k):
        self.k = k

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    


def main():
    xs_train, ys_train, xs_val, ys_val = get_training_data()

    # xs_train, ys_train, xs_val, ys_val = main_proc("train")

    xs_train = tf.convert_to_tensor(xs_train, dtype=tf.float32)
    ys_train = tf.convert_to_tensor(ys_train, dtype=tf.float32)
    xs_val = tf.convert_to_tensor(xs_val, dtype=tf.float32)
    ys_val = tf.convert_to_tensor(ys_val, dtype=tf.float32)

    print(xs_train)



if __name__ == '__main__':
    print("hello world")
    main()