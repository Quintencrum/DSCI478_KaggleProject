# KNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print('Hello ------------------ ?????????????')

class KNN:
    def __init__(self, k):
        self.k = k

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    


