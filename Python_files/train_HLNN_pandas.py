#~~ TRAIN_NN.PY ~~# 
# Will write this later

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import tensorflow as tf
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
batch_size  = 2000
X_train = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects2/X_train.pkl")
X_test = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects2/X_test.pkl")
y_train = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects2/y_train.pkl")
y_test = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects2/y_test.pkl")

# Combining input variables with flags


# prefetching a number of batches, to make loading more efficient possibly

# create model
def NN_creator(inputshape_3, denselayers, learningrate = 0.001):
    # Inputshape should be a 3-comp tuple, where 1st two els are height x width and 3rd is no. layers
    # conv/denselayers denote number of convolutional and dense layers in network
    # convlayers should be a tuple
    # dense necessarily come after convolutional
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



model = NN_creator((20,),5)
early_stop = EarlyStopping(monitor = 'val_loss', patience = 20)
model.compile(
    loss="mean_squared_error",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

history = History()

# fit model

time_start = time.time()
print("Training model")

model.fit(X_train, y_train,
          batch_size=2000,
          epochs=2000,
          callbacks=[history, early_stop],
          validation_data = (X_test, y_test),) 
model.save("/vols/cms/fjo18/Masters2021/Models/model_2000epoch")

time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
#print("splitting X into train/test...")
