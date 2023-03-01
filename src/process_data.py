from os import listdir
import numpy as np
import pandas as pd

import utilities as utils

RAW_DATA_FILES = {"test": "test.csv", "train": "train.csv"}
PROCESSED_DATA_FILES = {"xs_t": "xs_train.csv", "labels_t": "labels_train.csv",
                        "xs_v": "xs_validate.csv", "labels_v": "labels_validate.csv"}


def get_data(ftype: str):
    raw_df = pd.read_csv(utils.get_data_path().joinpath("raw",RAW_DATA_FILES[ftype]),
                     sep=',', header=[0], dtype=int)
    assert not np.any(raw_df.isna()), f"Nulls in the dataset {RAW_DATA_FILES[ftype]}"
    
    return raw_df.values

def get_training_data():
    process_dir = utils.get_data_path().joinpath("processed")
    
    xs_train = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["xs_t"]),
                           sep=',', header=[0], dtype='float32').values
    labels_train = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["labels_t"]),
                               sep=',', header=[0], dtype='float32').values.reshape((-1,))
    xs_validation = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["xs_v"]),
                                sep=',', header=[0], dtype='float32').values
    labels_validation = pd.read_csv(process_dir.joinpath(PROCESSED_DATA_FILES["labels_v"]),
                                    sep=',', header=[0], dtype='float32').values.reshape((-1,))

    return xs_train, labels_train, xs_validation, labels_validation

def main(ftype: str, validation_percent: float=0.2, overwrite=False):
    assert ftype in RAW_DATA_FILES.keys(), f"Choose data type from {RAW_DATA_FILES.keys()}"

    raw_df = get_data(ftype)

    if ftype == "test":
        return raw_df

    if overwrite:
        xs_train, labels_train, xs_validation, labels_validation = split_data(raw_df, validation_percent)

        # Loccally save the data files
        process_dir = utils.get_data_path().joinpath("processed")
        np.savetxt(process_dir.joinpath(PROCESSED_DATA_FILES["xs_t"]), xs_train, delimiter=',')
        np.savetxt(process_dir.joinpath(PROCESSED_DATA_FILES["labels_t"]), labels_train, delimiter=',')
        np.savetxt(process_dir.joinpath(PROCESSED_DATA_FILES["xs_v"]), xs_validation, delimiter=',')
        np.savetxt(process_dir.joinpath(PROCESSED_DATA_FILES["labels_v"]), labels_validation, delimiter=',')
        print("Locally saved the processed data")
        
        return xs_train, labels_train, xs_validation, labels_validation
    
    else:
        # Why was this "processed"? 
        process_dir = utils.get_data_path().joinpath("processed")

        # Check if files exist
        for file in listdir(process_dir):
            if file not in PROCESSED_DATA_FILES.values():
                return main(ftype, validation_percent, overwrite=True)
        if len(listdir(process_dir)) == 0:
            return main(ftype, validation_percent, overwrite=True)

        return get_training_data()
        

def split_data(raw_df: pd.DataFrame,
               validation_percent: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    assert 0<validation_percent<1, "Validation percentage should be between 0 and 1"
    df = raw_df.copy()
    validation_df = df.groupby('label').sample(frac=validation_percent, random_state=0)
    train_df = df.drop(validation_df.index)
    
    xs_train = train_df.iloc[:,1:].values.astype('float32') / 255.0
    labels_train = train_df.iloc[:,0].values.reshape((-1,))
    xs_validation = validation_df.iloc[:,1:].values.astype('float32') / 255.0
    labels_validation = validation_df.iloc[:,0].values.reshape((-1,))

    return xs_train, labels_train, xs_validation, labels_validation
