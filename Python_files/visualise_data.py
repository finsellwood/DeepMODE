#~~ VISUALISE_DATA.py ~~#
# This file loads in the numpy image arrays to ensure they're working properly
# Saves a figure with 20 images on (large + small side-by-side) for examination
rootpath = "/vols/cms/fjo18/Masters2021"
num_arrays = 73
use_dataset = True
# For choosing either the original (imgen.py) images or the final test/train set
plot_images = False
# saves time if doesn't save .png to cwd
test_x_shape = False
test_y_shape = False
test_full_df_shape = True

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#~~ Load pkl files ~~#
###
time_start = time.time()
print("loading in large image arrays...")
###
if use_dataset:
  l_image_array = np.load(rootpath + "/DataFrames/im_l_array_train.npy")
else:
  list_of_arrays = []
  for a in range(num_arrays):
    list_of_arrays.append(np.load(rootpath + "/Images/image_l_%02d.npy" % a))
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
  s_image_array = np.load(rootpath + "/DataFrames/im_s_array_train.npy")
else:
  list_of_arrays = []
  for a in range(num_arrays):
    list_of_arrays.append(np.load(rootpath + "/Images/image_s_%02d.npy" % a))
  s_image_array = np.concatenate(list_of_arrays)
  list_of_arrays = []
print('small arrays shape' + str(s_image_array.shape))

if plot_images:
  import random
  no_images = 10
  no_arrays = s_image_array.shape[0]
  fig, ax = plt.subplots(no_images,2)
  fig.set_size_inches(12, 40)
  for a in range(no_images):
    g = random.randint(0,no_arrays)

    ax[a][0].imshow(l_image_array[g], cmap='gray_r')
    ax[a][1].imshow(s_image_array[g], cmap='gray_r')
    
  plt.savefig('images.png', dpi = 500)

if test_x_shape:
  X_train = pd.read_pickle(rootpath + "/DataFrames/X_train_df.pkl")
  X_test = pd.read_pickle(rootpath + "/DataFrames/X_test_df.pkl")
  print("X shapes are", X_train.shape, X_test.shape)
  print(list(X_train.columns))
if test_y_shape:
  y_train = pd.read_pickle(rootpath + "/DataFrames/y_train_df.pkl")
  y_test = pd.read_pickle(rootpath + "/DataFrames/y_test_df.pkl")
  print("y shapes are", y_train.shape, y_test.shape)
if test_full_df_shape:
  df = pd.read_pickle(rootpath + "/Objects/ordereddf_modified.pkl")
  print("full_df shapes are", df.shape, df.shape)
  print(list(df.columns))

