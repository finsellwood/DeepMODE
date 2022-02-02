#~~ recover_model.py ~~# 
# For when the model is binned or cancelled but want to load from checkpoint
rootpath = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

model_source_folder = "/Models/"
# The location to load the sample model from. Must be IDENTICAL to the model that was lost
model_save_folder = "/Models_DM3/"
# Where to save the final model
data_folder = "/DataFrames/"
checkpoint_filepath = rootpath_save + "/Checkpoints/checkpoint"

all_decay_modes = False
no_modes = 3
if not all_decay_modes:
    model_source_folder = "/Models_DM2/"
    data_folder = "/DataFrames_DM/"

model_name = "LSH_model_0.718_20220128_125750"
model_path = rootpath + model_source_folder + model_name

use_inputs = [True, True, True]
use_unnormalised = True
drop_variables = False
# Initial parameters of the original model
small_dataset = False
small_dataset_size = 10000

import datetime
from math import ceil

# Load packages
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
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
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize, plot_model
import time
import pickle
# load data

###
time_start = time.time()
print("Loading test data")
###

y_train = pd.read_pickle(rootpath + data_folder + "y_train_df.pkl")
y_test = pd.read_pickle(rootpath + data_folder + "y_test_df.pkl")

l_im_test = []
s_im_test = []
X_test = pd.DataFrame()
# These need to be here so that the later operations don't break when you only use some inputs
if use_inputs[0]:
    #l_im_train = np.load(rootpath + "/DataFrames/im_l_array_train.npy")
    l_im_test = np.load(rootpath + data_folder + "im_l_array_test.npy")
if use_inputs[1]:
    #s_im_train = np.load(rootpath + "/DataFrames/im_s_array_train.npy")
    s_im_test = np.load(rootpath + data_folder + "im_s_array_test.npy")
if use_inputs[2]:
    if use_unnormalised:
        #X_train = pd.read_pickle(rootpath + "/DataFrames/X_n_train_df.pkl")
        X_test = pd.read_pickle(rootpath + data_folder + "X_test_df.pkl")
    else:
        #X_train = pd.read_pickle(rootpath + "/DataFrames/X_train_df.pkl")
        X_test = pd.read_pickle(rootpath + data_folder + "X_n_test_df.pkl")

if drop_variables:
    vars_to_drop = ['pi2_E_2', 'pi3_E_2','n_gammas_2','sc1_Nclusters_2','tau_E_2',]
    #X_train.drop(columns = vars_to_drop, inplace = True)
    X_test.drop(columns = vars_to_drop, inplace = True)

if small_dataset:
    test_size = int(small_dataset_size*.2)
    #train_size = int(small_dataset_size*.8)
    #X_train = X_train.head(train_size)
    X_test = X_test.head(test_size)
    #y_train = y_train[:train_size]
    y_test = y_test[:test_size]
    #l_im_train = l_im_train[:train_size]
    l_im_test = l_im_test[:test_size]
    #s_im_train = s_im_train[:train_size]
    s_im_test = s_im_test[:test_size]

test_full_inputs = [l_im_test, s_im_test, X_test]
test_inputs = []
for a in range(len(use_inputs)):
    if use_inputs[a]:
        test_inputs.append(test_full_inputs[a])
# Setting up test inputs based on the mask

# load model

###
time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("Loading model")
###

model = keras.models.load_model(model_path)
model.load_weights(checkpoint_filepath)

###
time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("Evaluating test dataset")
###

prediction = model.predict(test_inputs)
idx = prediction.argmax(axis=1)
y_pred = (idx[:,None] == np.arange(prediction.shape[1])).astype(float)
flatpred = np.argmax(y_pred, axis=-1)
flattest = np.argmax(y_test, axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

input_string = ''
inputflags = ['L', 'S', 'H']
for a in range(len(use_inputs)):
    if use_inputs[a]:
        input_string += inputflags[a]
model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(rootpath_save + model_save_folder + "%s_model_%.3f_%s" % (input_string, accuracy, model_datetime))#print(accuracy)
# print(confusion_matrix(flattest, flatpred, normalize = 'true'))
# normalize = 'true' gives EFFICIENCY
# print(confusion_matrix(flattest, flatpred, normalize = 'pred'))
# normalize = 'pred' gives PURITY

#print(classification_report(y_test, y_pred))
# Interesting metric here
#~~ Creating confusion arrays ~~#
