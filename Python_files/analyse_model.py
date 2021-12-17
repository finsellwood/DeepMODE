#~~ TRAIN_NN.PY ~~# 
# Will write this later
rootpath = "/vols/cms/fjo18/Masters2021"
model_name = "LS_model_0.556_20211217_155442"
model_path = rootpath + "/Models/" + model_name

use_inputs = [True, True, False]
use_unnormalised = True
drop_variables = False
# Initial parameters of the original model
small_dataset = False
small_dataset_size = 10000

plot_confusion_matrices = True

import datetime
from math import ceil
model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

# load data

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

# create model

model = keras.models.load_model(model_path)
prediction = model.predict(test_inputs)
idx = prediction.argmax(axis=1)
y_pred = (idx[:,None] == np.arange(prediction.shape[1])).astype(float)
# for a in range(50):
#     #print(y_pred[a], y_test[a])
#     print(y_train[a])

flatpred = np.argmax(y_pred, axis=-1)
#print(flatpred)
flattest = np.argmax(y_test, axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
#~~ Creating confusion arrays ~~#

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


    plt.savefig( model_path + '.png', dpi = 100)

