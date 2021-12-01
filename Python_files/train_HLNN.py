#~~ TRAIN_NN.PY ~~# 
# Will write this later

# Training parameters
batch_size  = 2000
no_dense_layers = 54
HL_shape = (20,)
stop_patience = 20
no_epochs = 5

import datetime
model_name = 'High_level_model' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

cache_dataset = True
save_model = False
training_parameters = [batch_size, no_dense_layers, stop_patience, no_epochs]


import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
# Code will now print the device on which it is running
from tensorflow import TensorSpec
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize
import time

# load data
X_train = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/X_train_tensor", element_spec = TensorSpec(shape=(20,), dtype=tf.float64, name=None)).batch(batch_size)
X_test = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/X_test_tensor", element_spec = TensorSpec(shape=(20,), dtype=tf.float64, name=None)).batch(batch_size)
y_train = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/y_train_tensor", element_spec = TensorSpec(shape=(6,), dtype=tf.float32, name=None)).batch(batch_size)
y_test = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/y_test_tensor", element_spec = TensorSpec(shape=(6,), dtype=tf.float32, name=None)).batch(batch_size)

train_batch = tf.data.Dataset.zip((X_train, y_train))
test_batch = tf.data.Dataset.zip((X_test, y_test))
# Combining input variables with flags

train_batch = train_batch.prefetch(tf.data.AUTOTUNE)
test_batch = test_batch.prefetch(tf.data.AUTOTUNE)
# prefetching a number of batches, to make loading more efficient possibly

if cache_dataset:
    train_batch = train_batch.cache()
    test_batch = test_batch.cache()
    print('caching dataset')
# cache dataset - runs quicker than without

# create model
def NN_creator(inputshape_3, denselayers, learningrate = 0.001):
    # Inputshape should be a 3-comp tuple, where 1st two els are height x width and 3rd is no. layers
    # denselayers denote number of dense layers in network
    no_dense_layers = denselayers
    input_hl = keras.Input(shape = inputshape_3)
    # Flatten output into 1D, so can be applied to dense layers more easily
    y = layers.Dense(88, activation= "relu", name = "hidden_0")(input_hl)
    x = layers.BatchNormalization(name = "BatchNorm_0")(y)
    for a in range(no_dense_layers - 1):
        y = layers.Dense(88, activation= "relu", name = "hidden_" + str(a+1))(x)
        x = layers.BatchNormalization(name = "BatchNorm_" + str(a+1))(y)
    
    outputs = layers.Dense(6, name = "outputs", activation = "softmax")(x)

    model = keras.Model(inputs= input_hl, outputs=outputs, name="NN_model")
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=learningrate), metrics=["accuracy"],)
    model.summary()
    return model


model = NN_creator(HL_shape, no_dense_layers)

early_stop = EarlyStopping(monitor = 'val_loss', patience = stop_patience)
history = History()

# fit model

time_start = time.time()
print("Training model")
print(training_parameters)

model.fit(train_batch,
#           batch_size=200,
          epochs=no_epochs,
          callbacks=[history, early_stop],
          validation_data = test_batch) 
if save_model:
    model.save("/vols/cms/fjo18/Masters2021/Models/%s" % model_name)

time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
#print("splitting X into train/test...")
