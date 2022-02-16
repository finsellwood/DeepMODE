#~~ VISUALISE_DATA.py ~~#
# This file loads in the numpy image arrays to ensure they're working properly
# Saves a figure with 20 images on (large + small side-by-side) for examination
rootpath_load = "/vols/cms/fjo18/Masters2021"
l_image_prefix = '/B_Images/Images3/m_image_l_'
s_image_prefix = '/B_Images/Images3/m_image_s_'
dataframe_prefix = "/C_DataFrames/DataFrames3_DM2"

no_layers = 7#6
num_arrays = 75#73
use_dataset = True
# For choosing either the original (imgen.py) images or the final test/train set

# analyse_images = True
# plot_images = True
analyse_images = True
plot_images = True
# saves time if doesn't save .png to cwd

test_x_shape = True
test_y_shape = False
test_full_df_shape = False
print_to_console = False
test_MVA = False

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
#~~ Load pkl files ~~#

if analyse_images:
  ###
  time_start = time.time()
  print("loading in large image arrays...")
  ###

  if use_dataset:
    l_image_array = np.load(rootpath_load + dataframe_prefix + "/im_l_array_train.npy")
  else:
    list_of_arrays = []
    for a in range(num_arrays):
      list_of_arrays.append(np.load(rootpath_load + l_image_prefix + "%02d.npy" % a))
    l_image_array = np.concatenate(list_of_arrays)
    list_of_arrays = []
  print('large arrays shape' + str(l_image_array.shape))

  ###
  time_elapsed = time.time() - time_start
  time_start = time.time()
  print("elapsed time = " + str(time_elapsed))
  print("loading in small image arrays...")
  ###
  if use_dataset:
    s_image_array = np.load(rootpath_load + dataframe_prefix + "/im_s_array_train.npy")
  else:
    list_of_arrays = []
    for a in range(num_arrays):
      list_of_arrays.append(np.load(rootpath_load + s_image_prefix + "%02d.npy" % a))
    s_image_array = np.concatenate(list_of_arrays)
    list_of_arrays = []
  print('small arrays shape' + str(s_image_array.shape))

  if plot_images:
    import random
    random.seed(12345)
    no_images = 10
    no_arrays = s_image_array.shape[0]
    fig, ax = plt.subplots(no_images,2 * no_layers)
    fig.set_size_inches(12*no_layers, 40)
    for a in range(no_images):
      g = random.randint(0,no_arrays)
      for b in range(no_layers):
        ax[a][b].imshow(l_image_array[g][:,:,b], cmap='gray_r')
        ax[a][b + no_layers].imshow(s_image_array[g][:,:,b], cmap='gray_r')
        
    plt.savefig('images.png', dpi = 500)

if test_x_shape:
  X_train = pd.read_pickle(rootpath_load + dataframe_prefix + "/X_train_df.pkl")
  X_test = pd.read_pickle(rootpath_load + dataframe_prefix + "/X_test_df.pkl")
  print("X shapes are", X_train.shape, X_test.shape)
  print(list(X_train.columns))
if test_y_shape:
  y_train = pd.read_pickle(rootpath_load + dataframe_prefix + "/y_train_df.pkl")
  y_test = pd.read_pickle(rootpath_load + dataframe_prefix + "/y_test_df.pkl")
  print("y shapes are", y_train.shape, y_test.shape)
if test_full_df_shape:
  df = pd.read_pickle(rootpath_load + "/Objects/ordereddf_modified.pkl")
  print("full_df shapes are", df.shape, df.shape)
  print(list(df.columns))

no = random.randint(0,1000)
if print_to_console:
  #print(l_image_array[no])
  print(s_image_array[107])
  print(no)
if test_MVA:
  count = 0
  mva_test = pd.read_pickle(rootpath_load + dataframe_prefix + "/mva_test.pkl")
  print(mva_test.shape)
  for  index, row in mva_test.iterrows():
    if row["tauFlag_2"] == -1:# and row["t_dm0raw_2"] == 0.0:
      count+=1
  print(count)
    # print(mva_test[["tauFlag_2", "t_dm0raw_2"]].sample(frac=1).head(100))
