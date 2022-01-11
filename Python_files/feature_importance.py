#~~ TRAIN_NN.PY ~~# 
# Will write this later
rootpath = "/vols/cms/fjo18/Masters2021"
model_name = "H_model_0.670_20211219_004316"
model_path = rootpath + "/Models/" + model_name

use_inputs = [True, True, True]
use_unnormalised = True
drop_variables = False
# Initial parameters of the original model
small_dataset = True
small_dataset_size = 100000

import datetime
from math import ceil

# Load packages
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
# Code will now print the device on which it is running
from tensorflow import TensorSpec
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras import layers
# from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Normalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize, plot_model
import eli5
from eli5.sklearn import PermutationImportance
import time
import pickle
# load data

###
time_start = time.time()
print("Loading test data")
###

y_train = pd.read_pickle(rootpath + "/DataFrames/y_train_df.pkl")
y_test = pd.read_pickle(rootpath + "/DataFrames/y_test_df.pkl")

l_im_test = []
s_im_test = []
X_test = pd.DataFrame()
# These need to be here so that the later operations don't break when you only use some inputs
if use_inputs[0]:
    #l_im_train = np.load(rootpath + "/DataFrames/im_l_array_train.npy")
    l_im_test = np.load(rootpath + "/DataFrames/im_l_array_test.npy")
if use_inputs[1]:
    #s_im_train = np.load(rootpath + "/DataFrames/im_s_array_train.npy")
    s_im_test = np.load(rootpath + "/DataFrames/im_s_array_test.npy")
if use_inputs[2]:
    if use_unnormalised:
        #X_train = pd.read_pickle(rootpath + "/DataFrames/X_n_train_df.pkl")
        X_test = pd.read_pickle(rootpath + "/DataFrames/X_n_test_df.pkl")
    else:
        #X_train = pd.read_pickle(rootpath + "/DataFrames/X_train_df.pkl")
        X_test = pd.read_pickle(rootpath + "/DataFrames/X_test_df.pkl")

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

###
time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("Evaluating test dataset")
###

# prediction = model.predict(test_inputs)
# idx = prediction.argmax(axis=1)
# y_pred = (idx[:,None] == np.arange(prediction.shape[1])).astype(float)
# # for a in range(50):s
# #     #print(y_pred[a], y_test[a])
# #     print(y_train[a])

# flatpred = np.argmax(y_pred, axis=-1)
# #print(flatpred)
# flattest = np.argmax(y_test, axis=-1)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

# def score(X, y):
#     y_pred = model.predict(X)
#     return accuracy_score(y, y_pred)

# base_score, score_decreases = eli5.permutation_importance.get_score_importances(score, test_inputs, y_test)
# feature_importances = np.mean(score_decreases, axis=0)

perm = PermutationImportance(model, scoring = 'accuracy', random_state=1).fit(test_inputs, y_test)
eli5.show_weights(perm, feature_names = test_inputs.columns.tolist())