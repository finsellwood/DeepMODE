#~~ IMGEN.PY ~~#
# Takes imvar_df dataframe and generates image files in batches of 100,000 events. Saves as numpy arrays.
# IMGEN_MULTI - adds multiple layers rather than just one for energy
load_path = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2_no_pi0"
save_path = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2_no_pi0"
# New image folder
no_events = 100000
#~~ Packages ~~#
import pandas as pd
import numpy as np
import vector
import awkward as ak  
import numba as nb
import time
#from sklearnr.externals import joblib
import pylab as pl


#~~ Load the dataframe with image variables in ~~#

print("Loading dataframes...")
time_start = time.time()

print("Loading data")
# I am going to load the training data into this object.
# This may be a bad idea
y_train = pd.read_pickle(load_path + "/y_train_df.pkl")
y_test = pd.read_pickle(load_path + "/y_test_df.pkl")
train_length = int(no_events*.8)
test_length = int(no_events*.2)
y_train = y_train[:train_length]
y_test = y_test[:test_length]
pd.to_pickle(y_train, save_path + "/y_train_small.pkl")
pd.to_pickle(y_test, save_path + "/y_test_small.pkl")
print(train_length)
print(test_length)

print("l_im_data")
l_im_train = np.load(load_path + "/im_l_array_train.npy")[:train_length]
pd.to_pickle(l_im_train, save_path + "/im_l_train_small.pkl")
del l_im_train
l_im_test = np.load(load_path + "/im_l_array_test.npy")[:test_length]
pd.to_pickle(l_im_test, save_path + "/im_l_test_small.pkl")
del l_im_test
print("s_im_data")
s_im_train = np.load(load_path + "/im_s_array_train.npy")[:train_length]
pd.to_pickle(s_im_train, save_path + "/im_s_train_small.pkl")
del s_im_train
s_im_test = np.load(load_path + "/im_s_array_test.npy")[:test_length]
pd.to_pickle(s_im_test, save_path + "/im_s_test_small.pkl")
del s_im_test
print("hl_data")
X_train = pd.read_pickle(load_path + "/X_train_df.pkl").head(train_length)
X_test = pd.read_pickle(load_path + "/X_test_df.pkl").head(test_length)
pd.to_pickle(X_train, save_path + "/X_train_small.pkl")
pd.to_pickle(X_test, save_path + "/X_test_small.pkl")

del X_train, X_test
print('done')

