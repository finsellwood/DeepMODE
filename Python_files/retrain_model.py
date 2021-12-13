#~~ retrain_model.PY ~~# 
# Takes an existing model and trains it again
rootpath = "/vols/cms/fjo18/Masters2021"

# Training parameters
batch_size = 2000 #1024
stop_patience = 20
no_epochs = 200
learningrate = 0.01

# Model architecture parameters
# # dense_layers = [(4,128, False), (2, 54, False)]
# dense_layers = [(4,22, False), (1, 128, False)]
# conv_layers = [(0,4), (0,3)]
# HL_shape = (21,)
# im_l_shape = (21,21,1)
# im_s_shape = (11,11,1)
# inc_dropout = True
# dropout_rate = [0.1, 0.5]
# 1st no. is conv and 2nd is dense
# Convolutional layers should have a much lower dropout rate than dense
use_inputs = [False, False, True]
model_path = rootpath + "/Models/Full_model_0.594_20211213_100715"
# A mask to check which inputs to use for the model - above indicates HL only


import datetime
from math import ceil
model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# cache_dataset = True
view_model = False
# DOESN'T WORK - HAVE TO INSTALL IF NECESSARY
save_model = False
small_dataset = False
small_dataset_size = 100000

training_parameters = [batch_size, use_inputs, learningrate, no_epochs, stop_patience, save_model, small_dataset]
training_parameter_names = ["batch size", \
    "inputs mask", "learning rate", "no. epochs", "stop patience", "save model?", "small dataset?"]

# Load packages
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
# Code will now print the device on which it is running
from tensorflow import TensorSpec
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Normalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize, plot_model

import time

# load data

y_train = pd.read_pickle(rootpath + "/DataFrames/y_train_df.pkl")
y_test = pd.read_pickle(rootpath + "/DataFrames/y_test_df.pkl")

l_im_train = []
l_im_test = []
s_im_train = []
s_im_test = []
X_train = []
X_test = []
# These need to be here so that the later operations don't break when you only use some inputs
if use_inputs[0]:
    l_im_train = np.load(rootpath + "/DataFrames/im_l_array_train.npy")
    l_im_test = np.load(rootpath + "/DataFrames/im_l_array_test.npy")
if use_inputs[1]:
    s_im_train = np.load(rootpath + "/DataFrames/im_s_array_train.npy")
    s_im_test = np.load(rootpath + "/DataFrames/im_s_array_test.npy")
if use_inputs[2]:
    X_train = pd.read_pickle(rootpath + "/DataFrames/X_train_df.pkl")
    X_test = pd.read_pickle(rootpath + "/DataFrames/X_test_df.pkl")




if small_dataset:
    test_size = int(small_dataset_size*.8)
    train_size = int(small_dataset_size*.2)
    X_train = X_train.head(test_size)
    X_test = X_test.head(train_size)
    y_train = y_train[:test_size]
    y_test = y_test[:train_size]
    l_im_train = l_im_train[:test_size]
    l_im_test = l_im_test[:train_size]
    s_im_train = s_im_train[:test_size]
    s_im_test = s_im_test[:train_size]


train_full_inputs = [l_im_train, s_im_train, X_train]
test_full_inputs = [l_im_test, s_im_test, X_test]
train_inputs = []
test_inputs = []
for a in range(len(use_inputs)):
    if use_inputs[a]:
        train_inputs.append(train_full_inputs[a])
        test_inputs.append(test_full_inputs[a])
# Setting up inputs based on the mask

# Load model
model = keras.models.load_model(model_path)
early_stop = EarlyStopping(monitor = 'val_loss', patience = stop_patience)
history = History()

# fit model
time_start = time.time()
print("Training model")


model.fit(train_inputs, y_train,
          batch_size = batch_size,
          epochs = no_epochs,
          callbacks=[history, early_stop],
          validation_data = (test_inputs, y_test)) 


prediction = model.predict(test_inputs)
idx = prediction.argmax(axis=1)
y_pred = (idx[:,None] == np.arange(prediction.shape[1])).astype(float)
flatpred = np.argmax(y_pred, axis=-1)
flattest = np.argmax(y_test, axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

if save_model:
    param_file = open(rootpath + "/Models/Retrained_model_%.3f_%s_params.txt" % (accuracy, model_datetime), 'w')
    model.save(rootpath + "/Models/Retrained_model_%.3f_%s" % (accuracy, model_datetime))
    for a in training_parameters:
        param_file.write(str(a) + '\n')
    param_file.write(accuracy + '\n')
    param_file.close()
    # Saves model parameters in a corresponding .txt file


time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))

