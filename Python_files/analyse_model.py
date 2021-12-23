#~~ TRAIN_NN.PY ~~# 
# Will write this later
rootpath = "/vols/cms/fjo18/Masters2021"
model_name = "LSH_model_0.687_20211222_141338"
model_path = rootpath + "/Models/" + model_name

use_inputs = [True, True, True]
use_unnormalised = True
drop_variables = False
# Initial parameters of the original model
small_dataset = False
small_dataset_size = 10000
plot_timeline = True

plot_confusion_matrices = True

import datetime
from math import ceil

# Load packages
import numpy as np
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
if plot_timeline:   
    history = pickle.load(open(model_path + '_history',  'rb'))
    # doesnt work i dont know why


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
#~~ Creating confusion arrays ~~#

###
time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("Plotting confusion matrices")
###

if plot_confusion_matrices:
    truelabels = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]) #for true modes 0,1,2,10,11,Other
    lengthstrue = [0,0,0,0,0,0]
    lengthspred = [0,0,0,0,0,0]
    for a in range(len(flattest)):
        truelabels[int(flattest[a])][int(flatpred[a])] +=1
        lengthstrue[int(flattest[a])] +=1
        lengthspred[int(flatpred[a])] +=1
    truelabelpurity = truelabels/lengthspred
    truelabelefficiency = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]], dtype = float)
    for a in range(6):
        for b in range(6):
            truelabelefficiency[a][b] = truelabels[a][b]/lengthstrue[a]

    plt.rcParams.update({'figure.autolayout': True})
    labellist = [r'$\pi^{\pm}$', r'$\pi^{\pm} \pi^0$', r'$\pi^{\pm} 2\pi^0$', r'$3\pi^{\pm}$', r'$3\pi^{\pm} \pi^0$', 'other']
    fig, ax = plt.subplots(1,2)
    plt.tight_layout()
    fig.set_size_inches(12, 8)

    ax[0].imshow(truelabelefficiency, cmap = 'Blues')
    for i in range(6):
        for j in range(6):
            if truelabelefficiency[i, j] > 0.5:
                text = ax[0].text(j, i, round(truelabelefficiency[i, j], 3),
                            ha="center", va="center", color="w")
            else:
                text = ax[0].text(j, i, round(truelabelefficiency[i, j], 3),
                            ha="center", va="center", color="black")

            
    ax[0].set_title('Efficiency')
    ax[0].set_xticks([0,1,2,3,4,5])
    ax[0].set_yticks([0,1,2,3,4,5])
    ax[0].set_xticklabels(labellist)
    ax[0].set_yticklabels(labellist)
    ax[0].set_xlabel('Predicted Mode')
    ax[0].set_ylabel('True Mode')


    ax[1].imshow(truelabelpurity, cmap = 'Blues')
    for i in range(6):
        for j in range(6):
            if truelabelpurity[i, j] > 0.5:
                text = ax[1].text(j, i, round(truelabelpurity[i, j], 3),
                            ha="center", va="center", color="w")
            else:
                text = ax[1].text(j, i, round(truelabelpurity[i, j], 3),
                            ha="center", va="center", color="black")

    ax[1].set_title('Purity')
    ax[1].set_xticks([0,1,2,3,4,5])
    ax[1].set_yticks([0,1,2,3,4,5])
    ax[1].set_xticklabels(labellist)
    ax[1].set_yticklabels(labellist)
    ax[1].set_xlabel('Predicted Mode')
    ax[1].set_ylabel('True Mode')


    plt.savefig( model_path + '_cm_' + '.png', dpi = 100)

###
time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("Plotting timeline")
###

if plot_timeline:
    # Extract number of run epochs from the training history
    epochs = range(1, len(history["loss"])+1)
    fig, ax = plt.subplots(2,1)
    # Extract loss on training and validation ddataset and plot them together
    ax[0].plot(epochs, history["loss"], "o-", label="Training")
    ax[0].plot(epochs, history["val_loss"], "o-", label="Test")
    ax[0].set_xlabel("Epochs"), ax[0].set_ylabel("Loss")
    ax[0].set_yscale("log")
    ax[0].legend()

    # do the same for the accuracy:
    # Extract number of run epochs from the training history
    epochs2 = range(1, len(history["accuracy"])+1)

    # Extract accuracy on training and validation ddataset and plot them together
    ax[1].plot(epochs2, history["accuracy"], "o-", label="Training")
    ax[1].plot(epochs2, history["val_accuracy"], "o-", label="Test")
    ax[1].set_xlabel("Epochs"), ax[1].set_ylabel("accuracy")
    ax[1].legend()
    
    plt.savefig( model_path + '_tl_' + '.png', dpi = 100)


