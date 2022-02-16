#~~ f_remove_pi0_data.py ~~#
# Get rid of elements that use pi0 reconstruction, see what happens
# Repurposing split_by_decay_mode, some names are outdated

rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

dataframe_prefix = "/C_DataFrames/DataFrames3_DM2"
# Folder to load dataframes from
data_folder = "/C_DataFrames/DataFrames3_DM2_no_pi0/"
# Folder to save new dataframes to


debug = False
check = False
debug_frac = 0.001

import numpy as np
import pandas as pd
# from Python_files.C_Analysis.analyse_model import MVA_values
# Where did this line come from????? I didn't write this
import time
import matplotlib.pyplot as plt



#~~ Load pkl files ~~#
time_start = time.time()
print("loading x files...")
###


x_train = pd.read_pickle(rootpath_load + dataframe_prefix + "/X_train_df.pkl")
x_test = pd.read_pickle(rootpath_load + dataframe_prefix + "/X_test_df.pkl")

no_train_events = x_train.shape[0]
no_test_events = x_test.shape[0]

if debug:
      no_train_events = int(no_train_events * debug_frac)
      no_test_events = int(no_test_events * debug_frac)
      x_train = x_train.head(no_train_events).reset_index(drop=True)
      x_test = x_test.head(no_test_events).reset_index(drop=True)


### Remove pi0-related HL variables ###
cols_to_remove = ['pi0_E_2', 'pi0_2mass','E_pi0/E_tau']
x_train.drop(columns = cols_to_remove, inplace = True)
x_test.drop(columns = cols_to_remove, inplace = True)

print(x_train.shape)
print(list(x_train.columns))

pd.to_pickle(x_train, rootpath_save + data_folder + "X_train_df.pkl")
pd.to_pickle(x_test, rootpath_save + data_folder + "X_test_df.pkl")

del x_train,x_test

      
print('Doing images')
# Create numpy arrays for simple iteration
im_l_array_train = np.load(rootpath_load + dataframe_prefix + "/im_l_array_train.npy")[:no_train_events]
im_l_array_test = np.load(rootpath_load + dataframe_prefix + "/im_l_array_test.npy")[:no_test_events]
im_s_array_train = np.load(rootpath_load + dataframe_prefix + "/im_s_array_train.npy")[:no_train_events]
im_s_array_test = np.load(rootpath_load + dataframe_prefix + "/im_s_array_test.npy")[:no_test_events]

layer_mask = [True,True,True,False,True,True,True]

no_pi0_l_train = im_l_array_train[:,:,:,layer_mask]
no_pi0_l_test = im_l_array_test[:,:,:,layer_mask]

no_pi0_s_train = im_s_array_train[:,:,:,layer_mask]
no_pi0_s_test = im_s_array_test[:,:,:,layer_mask]

if check:
    import random
    random.seed(12345)
    no_images = 2
    no_arrays = im_s_array_train.shape[0]
    no_layers_orig = im_s_array_train.shape[3]
    no_layers_new = no_pi0_s_train.shape[3] 

    fig, ax = plt.subplots(no_images, no_layers_orig + no_layers_new)
    fig.set_size_inches(6*no_layers_orig, 40)

    for a in range(no_images):
      g = random.randint(0,no_arrays)
      for b in range(no_layers_orig):
            ax[a][b].imshow(im_s_array_train[g][:,:,b], cmap='gray_r')
      for b in range(no_layers_new):
            ax[a][b + no_layers_orig].imshow(no_pi0_s_train[g][:,:,b], cmap='gray_r')
        
    plt.savefig('images.png', dpi = 500)

np.save(rootpath_save + data_folder + "im_l_array_train.npy", no_pi0_l_train)
np.save(rootpath_save + data_folder + "im_l_array_test.npy", no_pi0_l_test)
np.save(rootpath_save + data_folder + "im_s_array_train.npy", no_pi0_s_train)
np.save(rootpath_save + data_folder + "im_s_array_test.npy", no_pi0_s_test)

print("END")
