#~~ split_by_decay_mode.py ~~#
# Divide initial dataframes by decay mode
# Can change weighting with dataset_weights and use_as_mask, which keeps original ratios 
# But only returns modes with nonzero weights
# E.g. only 1pr in original ratio would use dataset_weights = [1,1,1,0,0,0], use_as_mask = True
rootpath = "/vols/cms/fjo18/Masters2021"
data_folder = "/DataFrames_DM2/"
split_X = True
split_MVA = False
split_images = True
no_categories = 6

debug = False
debug_frac = 0.001

dataset_weights = [1,1,1,0,0,0]
# Weights of the different decay modes (pi, pi+pi0, pi+2pi0, 3pi, 3pi+pi0, other)
dataset_weights = [a/sum(dataset_weights) for a in dataset_weights]
# Normalise weights to one
use_as_mask = False
mask = [a != 0 for a in dataset_weights]

no_categories = sum(mask)
print(no_categories, 'ouput categories')

import numpy as np
import pandas as pd
from tensorflow import keras
import time


#~~ Load pkl files ~~#
time_start = time.time()
print("loading y,x files...")
###


y_train = pd.read_pickle(rootpath + "/DataFrames/y_train_df.pkl")
y_test = pd.read_pickle(rootpath + "/DataFrames/y_test_df.pkl")
y_train_flat = pd.DataFrame(np.argmax(y_train, axis=-1), columns = ["tauFlag_2"])
y_test_flat = pd.DataFrame(np.argmax(y_test, axis=-1), columns = ["tauFlag_2"])

no_train_events = y_train.shape[0]
no_test_events = y_test.shape[0]

del y_test, y_train

if debug:
      no_train_events = int(no_train_events * debug_frac)
      no_test_events = int(no_test_events * debug_frac)

      y_train_flat = y_train_flat.head(no_train_events)
      y_test_flat = y_test_flat.head(no_test_events)      

if split_X:
      print('Splitting X')
      x_train = pd.read_pickle(rootpath + "/DataFrames/X_n_train_df.pkl").head(no_train_events).reset_index(drop=True)
      x_test = pd.read_pickle(rootpath + "/DataFrames/X_n_test_df.pkl").head(no_test_events).reset_index(drop=True)

      x_train_concat = pd.concat([x_train, y_train_flat], axis = 1, join='inner')
      x_test_concat = pd.concat([x_test, y_test_flat], axis = 1, join='inner')

      del x_train, x_test, y_train_flat, y_test_flat

      ### Split train into DMs ###
      x_train_DM0 = x_train_concat[
      (x_train_concat["tauFlag_2"] == 0)
      ]
      x_train_DM1 = x_train_concat[
      (x_train_concat["tauFlag_2"] == 1)
      ]
      x_train_DM2 = x_train_concat[
      (x_train_concat["tauFlag_2"] == 2)
      ]
      x_train_DM10 = x_train_concat[
      (x_train_concat["tauFlag_2"] == 3)
      ]
      x_train_DM11 = x_train_concat[
      (x_train_concat["tauFlag_2"] == 4)
      ]
      x_train_DMminus1 = x_train_concat[
      (x_train_concat["tauFlag_2"] == 5)
      ]

      ### Split test into DMs ###
      x_test_DM0 = x_test_concat[
      (x_test_concat["tauFlag_2"] == 0)
      ]
      x_test_DM1 = x_test_concat[
      (x_test_concat["tauFlag_2"] == 1)
      ]
      x_test_DM2 = x_test_concat[
      (x_test_concat["tauFlag_2"] == 2)
      ]
      x_test_DM10 = x_test_concat[
      (x_test_concat["tauFlag_2"] == 3)
      ]
      x_test_DM11 = x_test_concat[
      (x_test_concat["tauFlag_2"] == 4)
      ]
      x_test_DMminus1 = x_test_concat[
      (x_test_concat["tauFlag_2"] == 5)
      ]

      del x_train_concat, x_test_concat

      trainlist = [x_train_DM0,x_train_DM1,x_train_DM2,x_train_DM10,x_train_DM11,x_train_DMminus1]
      trainshapes = [a.shape[0] for a in trainlist]
      testlist = [x_test_DM0,x_test_DM1,x_test_DM2,x_test_DM10,x_test_DM11,x_test_DMminus1]
      testshapes = [a.shape[0] for a in testlist]

      min_train = no_train_events
      min_train_index = 0
      for a in range(len(dataset_weights)):
            if dataset_weights[a] != 0.0:
                  no_things = (trainshapes[a]/dataset_weights[a])
                  if no_things < min_train:
                        min_train = no_things
                        min_train_index = a
      new_train_size = min_train
      new_test_size = testshapes[min_train_index]/dataset_weights[min_train_index]
      if use_as_mask:
            train_lengths = [a*b for a,b in zip(trainshapes, mask)]
            test_lengths = [a*b for a,b in zip(testshapes, mask)]
      else:
            train_lengths = [int(a * new_train_size) for a in dataset_weights]
            test_lengths = [int(a * new_test_size) for a in dataset_weights]
      print(train_lengths)
      print(test_lengths)
      for a in range(len(dataset_weights)):
            trainlist[a] = trainlist[a].head(train_lengths[a])
            testlist[a] = testlist[a].head(test_lengths[a])

      train_dataset = pd.concat(trainlist)
      test_dataset = pd.concat(testlist)
      
      
      # for a in range(len(trainlist)):
      #       print(a)
      #       print(trainlist[a])
      #       del trainlist[a], testlist[a]
      # Doesnt work - list isn't as long as it thinks it is since some els are empty

      train_dataset = train_dataset.sample(frac=1, random_state=12345)#.reset_index(drop=True)
      train_index = train_dataset.index
      train_dataset.reset_index(drop=True)
      test_dataset = test_dataset.sample(frac=1, random_state=12345)#.reset_index(drop=True)
      test_index = test_dataset.index
      test_dataset.reset_index(drop=True)

      # Split the flags from everything else
      y_train_split = train_dataset['tauFlag_2']
      y_test_split = test_dataset['tauFlag_2']
      train_dataset.drop(columns = 'tauFlag_2', inplace = True)
      test_dataset.drop(columns = 'tauFlag_2', inplace = True)
      print(y_train_split.head())

      g, y_train_indices = np.unique(y_train_split, return_inverse=True)
      g, y_test_indices = np.unique(y_test_split, return_inverse=True)
      # g is thrown away, returns the *order* of y-values rather than exact values (i.e. [1,3,5] returns [0,1,2])
      y_train_split = keras.utils.to_categorical(y_train_indices, no_categories)
      y_test_split = keras.utils.to_categorical(y_test_indices, no_categories)
      # Convert y arrays back to one-hot form

      pd.to_pickle(train_dataset, rootpath + data_folder + "X_train_df.pkl")
      pd.to_pickle(test_dataset, rootpath + data_folder + "X_test_df.pkl")
      pd.to_pickle(y_train_split, rootpath + data_folder + "y_train_df.pkl")
      pd.to_pickle(y_test_split, rootpath + data_folder + "y_test_df.pkl")
      print(train_dataset.shape, 'X_train shape')
      print(test_dataset.shape, 'X_test shape')
      print(y_train_split.shape, 'y_train shape')
      print(y_test_split.shape, 'y_test shape')
      del train_dataset, test_dataset, y_train_split, y_test_split
      

if split_images:
      print('Splitting images')
      # Create numpy arrays for simple iteration
      im_l_array_train = np.load(rootpath + "/DataFrames/im_l_array_train.npy")[:no_train_events]
      im_l_array_test = np.load(rootpath + "/DataFrames/im_l_array_test.npy")[:no_test_events]
      im_s_array_train = np.load(rootpath + "/DataFrames/im_s_array_train.npy")[:no_train_events]
      im_s_array_test = np.load(rootpath + "/DataFrames/im_s_array_test.npy")[:no_test_events]

      im_l_train_dataset = im_l_array_train[train_index]
      im_l_test_dataset = im_l_array_test[test_index]
      im_s_train_dataset = im_s_array_train[train_index]
      im_s_test_dataset = im_s_array_test[test_index] 

      print(im_l_train_dataset.shape, 'im_l_train shape')
      print(im_l_test_dataset.shape, 'im_l_test shape')
      print(im_s_train_dataset.shape, 'im_l_train shape')
      print(im_s_test_dataset.shape, 'im_l_test shape')

      np.save(rootpath + data_folder + "im_l_array_train.npy", im_l_train_dataset)
      np.save(rootpath + data_folder + "im_l_array_test.npy", im_l_test_dataset)
      np.save(rootpath + data_folder + "im_s_array_train.npy", im_s_train_dataset)
      np.save(rootpath + data_folder + "im_s_array_test.npy", im_s_test_dataset)



print("END")