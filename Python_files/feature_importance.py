#~~ TRAIN_NN.PY ~~# 
# Will write this later
rootpath = "/vols/cms/fjo18/Masters2021"
model_name = "LSH_model_0.692_20211218_194420"
model_path = rootpath + "/Models/" + model_name

use_inputs = [True, True, True]
use_unnormalised = True
drop_variables = False
# Initial parameters of the original model
small_dataset = True
small_dataset_size = 100000

plot_EP = True

import datetime
from math import ceil

# Load packages
import numpy as np
from numpy.lib.twodim_base import diag
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
import eli5
from eli5.sklearn import PermutationImportance
import time
import pickle
# load data
from sklearn.inspection import permutation_importance

###
time_start = time.time()
print("Loading test data")
###

y_train = pd.read_pickle(rootpath + "/DataFrames/y_train_df.pkl")
y_test = pd.read_pickle(rootpath + "/DataFrames/y_test_df.pkl")

mva_train = pd.read_pickle(rootpath + "/Objects/mva_train.pkl")
mva_test = pd.read_pickle(rootpath + "/Objects/mva_test.pkl")


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

prediction = model.predict(test_inputs)
idx = prediction.argmax(axis=1)
y_pred = (idx[:,None] == np.arange(prediction.shape[1])).astype(float)
# for a in range(50):s
#     #print(y_pred[a], y_test[a])
#     print(y_train[a])

flatpred = np.argmax(y_pred, axis=-1)
#print(flatpred)
flattest = np.argmax(y_test, axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

HPS_values = test_inputs[-1]["tau_decay_mode_2"]
MVA_values = mva_test["mva_dm_2"]

dm_indices = [0,1,2,10,11,-1]
dm_output_indices = [0,1,2,3,4,5]
def remapper(function, inputs, outputs):
    output = (function == inputs[0]) * outputs[0] + (function == inputs[1]) * outputs[1] + \
    (function == inputs[2]) * outputs[2] + (function == inputs[3]) * outputs[3] + \
    (function == inputs[4]) * outputs[4] + (function == inputs[5]) * outputs[5]
    return output 

transformed_HPS = remapper(HPS_values, dm_indices, dm_output_indices)
transformed_MVA = remapper(MVA_values, dm_indices, dm_output_indices)
transformed_HPS = transformed_HPS.to_numpy()
transformed_MVA = transformed_MVA.to_numpy()
print(transformed_HPS, transformed_MVA)

# HPS data but with the same values as the fitted function
if plot_EP:    
    diag_elements_NN = np.array([0,0,0,0,0,0])
    diag_elements_HPS = np.array([0,0,0,0,0,0])
    diag_elements_MVA = np.array([0,0,0,0,0,0])


    lengthstrue = np.array([0,0,0,0,0,0])
    lengthspred_NN = np.array([0,0,0,0,0,0])
    lengthspred_HPS = np.array([0,0,0,0,0,0])
    lengthspred_MVA = np.array([0,0,0,0,0,0])

    for a in range(len(flattest)):
        if flattest[a] == flatpred[a]:
            diag_elements_NN[int(flattest[a])] +=1
        if flattest[a] == transformed_MVA[a]:
            diag_elements_MVA[int(flattest[a])] +=1
        if flattest[a] == transformed_HPS[a]:
            diag_elements_HPS[int(flattest[a])] +=1
        
        # truelabels[int(flattest[a])][int(flatpred[a])] +=1
        lengthstrue[int(flattest[a])] +=1
        lengthspred_NN[int(flatpred[a])] +=1
        lengthspred_HPS[int(transformed_HPS[a])] +=1
        lengthspred_MVA[int(transformed_MVA[a])] +=1


    diag_purity_NN = diag_elements_NN/lengthspred_NN
    diag_purity_HPS = diag_elements_HPS/lengthspred_HPS
    diag_purity_MVA = diag_elements_MVA/lengthspred_MVA

    # Purity is /by reconstructed lengths
    diag_efficiency_NN = diag_elements_NN/lengthstrue
    diag_efficiency_HPS = diag_elements_HPS/lengthstrue
    diag_efficiency_MVA = diag_elements_MVA/lengthstrue

    # Efficiency is /by true lengths
    print(diag_purity_NN, diag_purity_MVA, diag_purity_HPS)
    print(diag_efficiency_NN, diag_efficiency_MVA, diag_efficiency_HPS)
    
    fig, ax = plt.subplots(1,2)
    plt.rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    labellist = [r'$\pi^{\pm}$', r'$\pi^{\pm} \pi^0$', r'$\pi^{\pm} 2\pi^0$', r'$3\pi^{\pm}$', r'$3\pi^{\pm} \pi^0$', 'other']
    fig.set_size_inches(12,8)
    indices = np.array([0,1,2,3,4,5])
    
    ax[1].bar(indices - 0.2, diag_efficiency_NN, width = 0.2)
    ax[1].bar(indices, diag_efficiency_MVA, width = 0.2)
    ax[1].bar(indices + 0.2, diag_efficiency_HPS, width = 0.2)
    
    
    ax[1].set_ylim([0,1])
    ax[1].set_title('Efficiency')
    ax[1].set_xticks([0,1,2,3,4,5])
    ax[1].set_xticklabels(labellist)
    ax[1].set_xlabel('Predicted Mode')
    ax[1].set_ylabel('Efficiency')

    ax[0].bar(indices - 0.2, diag_purity_NN, width = 0.2)
    ax[0].bar(indices, diag_purity_MVA, width = 0.2)
    ax[0].bar(indices + 0.2, diag_purity_HPS, width = 0.2)

    ax[0].set_ylim([0,1])
    ax[0].set_title('Purity')
    ax[0].set_xticks([0,1,2,3,4,5])
    ax[0].set_xticklabels(labellist)
    ax[0].set_xlabel('Predicted Mode')
    ax[0].set_ylabel('Purity')
    for a in range(len(indices)):
        ax[0].text(indices[a] - 0.2, diag_purity_NN[a]+0.02, round(diag_purity_NN[a], 2), 
        ha="center", va="center", color = 'black')
        ax[0].text(indices[a], diag_purity_MVA[a]+0.02, round(diag_purity_MVA[a], 2), 
        ha="center", va="center", color = 'black')
        ax[0].text(indices[a] + 0.2, diag_purity_HPS[a]+0.02, round(diag_purity_HPS[a], 2), 
        ha="center", va="center", color = 'black')
    
        ax[1].text(indices[a] - 0.2, diag_efficiency_NN[a]+0.02, round(diag_efficiency_NN[a], 2), 
        ha="center", va="center", color = 'black')
        ax[1].text(indices[a], diag_efficiency_MVA[a]+0.02, round(diag_efficiency_MVA[a], 2), 
        ha="center", va="center", color = 'black')
        ax[1].text(indices[a] + 0.2, diag_efficiency_HPS[a]+0.02, round(diag_efficiency_HPS[a], 2), 
        ha="center", va="center", color = 'black')
    plt.savefig( model_path + '_NNvsHPS_' + '.png', dpi = 100)

# def score(X, y):
#     y_pred = model.predict(X)
#     return accuracy_score(y, y_pred)

# base_score, score_decreases = eli5.permutation_importance.get_score_importances(score, test_inputs, y_test)
# feature_importances = np.mean(score_decreases, axis=0)

# perm = PermutationImportance(model, scoring = 'accuracy', random_state=1).fit(test_inputs, y_test)
# # eli5.show_weights(perm, feature_names = test_inputs.columns.tolist())
# featurenames = ['L_images', 'S_images'] + test_inputs[-1].columns.tolist()
# print(featurenames)
# eli5.show_weights(perm, feature_names = featurenames)


