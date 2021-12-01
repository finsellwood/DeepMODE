#~~ CREATE_DATASET.PY ~~#
# Loads in and splits all the existing data (in 'objects') into training and testing splits
# Hopefully can save images in only two .sav files
# WORKFLOW: 
# 1   LOAD IN y_ordederd (needs to be here all the time)
# 2   Load in df_ordered (modified), and drop unnecessary columns
#       Split df_ordered into train/test data (stratify with y_ordered)
#       convert train/test data to tensors with tf.convert_to_tensor(PANDAS DATAFRAME)
#       Save these files with tf.data.experimental.save (can load later with tf.data.experimental.load) into some useful location
#       Remove df_ordered from memory entirely
# 3   Load in the image files (numpy arrays)
#       Concatenate into one large array
#      Apply train_test_split with same random state, stratify with y. Maybe split y as well
#      Convert to tensorflow tensor as above
#      Save as above
#
# OUTPUTS: should have 6 tensorflow files, 3 train and 3 test each for HL vars, Images and y values

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


#~~ Load pkl files ~~#
time_start = time.time()
print("loading y,x files...")

y = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects/yvalues.pkl")
df_ordered = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects/ordereddf_modified.pkl")

time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("splitting X into train/test...")

X = df_ordered.drop(["tauFlag_2", 
                # Generator-level properties, actual decay mode of taus for training
                #"pi_px_2", "pi_py_2", "pi_pz_2", #"pi_E_2", 
                #"pi2_px_2", "pi2_py_2", "pi2_pz_2", #"pi2_E_2",
                #"pi3_px_2", "pi3_py_2", "pi3_pz_2", #"pi3_E_2",
                # 4-momenta of the charged pions
                # Note: pi2/pi3 only apply for 3pr modes
                "pi0_px_2", "pi0_py_2", "pi0_pz_2", #"pi0_E_2", 
                # 4-momenta of neutral pions
                "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                # 4-momenta of two leading photons
                "n_gammas_2", #"gam_px_2", "gam_py_2", "gam_pz_2", 
                # 3-momenta vectors of all photons
                #"sc1_px_2", "sc1_py_2", "sc1_pz_2",# "sc1_E_2",
                # 4-momentum of the supercluster
                #"cl_px_2", "cl_py_2", "cl_pz_2", "sc1_Nclusters_2", 
                # 3-momenta of clusters in supercluster
                "tau_px_2", "tau_py_2", "tau_pz_2", "tau_E_2",
                # 4-momenta of 'visible' tau
                #"tau_decay_mode_2", 
                # HPS algorithm decay mode
                "pt_2",
           ], axis=1).reset_index(drop=True)

del df_ordered
# print(X.columns.values.tolist())
# print(X.head())



X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=123456,
    stratify = y
)

del X, y

y_train = keras.utils.to_categorical(y_train, 6)
y_test = keras.utils.to_categorical(y_test, 6)

time_elapsed = time.time() - time_start
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("Saving X, y files...")


pd.to_pickle(X_train, "/vols/cms/fjo18/Masters2021/Objects2/X_train.pkl")
pd.to_pickle(X_test, "/vols/cms/fjo18/Masters2021/Objects2/X_test.pkl")
pd.to_pickle(y_train, "/vols/cms/fjo18/Masters2021/Objects2/y_train.pkl")
pd.to_pickle(y_test, "/vols/cms/fjo18/Masters2021/Objects2/y_test.pkl")


time_elapsed = time.time() - time_start
time_start = time.time()
print("elapsed time = " + str(time_elapsed))


print("END")
