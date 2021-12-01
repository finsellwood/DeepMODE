import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

new_dataset = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/X_train_tensor")
for row in new_dataset.take(3):
  print(row)
