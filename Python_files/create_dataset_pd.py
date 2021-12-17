#~~ CREATE_DATASET_pd.PY ~~#
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
#      Concatenate into one large array
#      Apply train_test_split with same random state, stratify with y. Maybe split y as well
#      Convert to tensorflow tensor as above
#      Save as above
#
# OUTPUTS: should have 6 tensorflow files, 3 train and 3 test each for HL vars, Images and y values
# EDIT - this version of the document has the option to create pandas dataframes or tf.tensors
# Also a file 'element_specs.txt' which contains the datatype of each tensor, for use when loading in the tensors later
rootpath = "/vols/cms/fjo18/Masters2021"
frames = True
sets = False
num_arrays = 73


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


#~~ Load pkl files ~~#
time_start = time.time()
print("loading y,x files...")
###

specfile = open('element_specs.txt', 'w')
y = pd.read_pickle(rootpath + "/Objects/yvalues.pkl")
df_ordered = pd.read_pickle(rootpath + "/Objects/ordereddf_modified.pkl")

###
time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("splitting X into train/test...")
###

X = df_ordered.drop(["tauFlag_2", 
                # Generator-level properties, actual decay mode of taus for training
                #"pi_px_2", "pi_py_2", "pi_pz_2", #"pi_E_2", 
                #"pi2_px_2", "pi2_py_2", "pi2_pz_2", #"pi2_E_2",
                #"pi3_px_2", "pi3_py_2", "pi3_pz_2", #"pi3_E_2",
                # 4-momenta of the charged pions
                # Note: pi2/pi3 only apply for 3pr modes
                # "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2", 
                # 4-momenta of neutral pions
                "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                # 4-momenta of two leading photons
                #"n_gammas_2", #"gam_px_2", "gam_py_2", "gam_pz_2", 
                # 3-momenta vectors of all photons
                #"sc1_px_2", "sc1_py_2", "sc1_pz_2",# "sc1_E_2",
                # 4-momentum of the supercluster
                #"cl_px_2", "cl_py_2", "cl_pz_2", "sc1_Nclusters_2", 
                # 3-momenta of clusters in supercluster
                "tau_px_2", "tau_py_2", "tau_pz_2", #"tau_E_2", 
                # 4-momenta of 'visible' tau
                #"tau_decay_mode_2", 
                # HPS algorithm decay mode
                #"pt_2",
           ], axis=1).reset_index(drop=True)

del df_ordered
X_not_norm = X
# Only normalise part of the dataframe (exclude discrete elements)
cols_not_to_normalise = ["n_gammas_2", "sc1_Nclusters_2", "tau_decay_mode_2",]
discrete_X = X[cols_not_to_normalise]
# discrete_X stays as dataframe, not normalised
to_norm_X = X.drop(cols_not_to_normalise, axis=1).reset_index(drop=True)
col_labels = list(to_norm_X.columns)
to_norm_X.to_numpy()
normX = normalize(to_norm_X)
# Convert to_norm_X into numpy array so that can use sklearn.normalize() on it
norm_df = pd.DataFrame(data = normX, columns = col_labels)
X = pd.concat([norm_df,discrete_X], axis = 1)
# Convert back to dataframes and sandwich two dfs back together
# Want as dataframes so that can see column names 

X_train, X_test, X_n_train, X_n_test = train_test_split(
    X, X_not_norm,
    test_size=0.2,
    random_state=123456,
    stratify = y
)

del X

###
time_elapsed = time.time() - time_start
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
###

if sets:
  print("Creating X tensor files...")
  t_X_train = tf.data.Dataset.from_tensor_slices(X_train)
  t_X_test = tf.data.Dataset.from_tensor_slices(X_test)

  specfile.write(str(t_X_train.element_spec) + '\n')
  specfile.write(str(t_X_test.element_spec) + '\n')

  print('Saving tensors...')
  tf.data.experimental.save(t_X_train, rootpath + "/Tensors/X_train_tensor")
  tf.data.experimental.save(t_X_test, rootpath + "/Tensors/X_test_tensor")
  del t_X_train, t_X_test

if frames:
  print("Saving X pd files...")
  pd.to_pickle(X_train, rootpath + "/DataFrames/X_train_df.pkl")
  pd.to_pickle(X_test, rootpath + "/DataFrames/X_test_df.pkl")
  pd.to_pickle(X_n_train, rootpath + "/DataFrames/X_n_train_df.pkl")
  pd.to_pickle(X_n_test, rootpath + "/DataFrames/X_n_test_df.pkl")

del X_train, X_test

###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# print("loading in large image arrays...")
# ###

# list_of_arrays = []
# for a in range(num_arrays):
#   list_of_arrays.append(np.load(rootpath + "/Images/image_l_%02d.npy" % a))
# l_image_array = np.concatenate(list_of_arrays)
# list_of_arrays = []

# ###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# print("Split large images for train/test...")
# ###

# im_l_array_train, im_l_array_test = train_test_split(
#     l_image_array,
#     test_size=0.2,
#     random_state=123456,
#     stratify = y)

# del l_image_array

# ###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# ###

# if sets:
#   print("Converting large image arrays to tensor files...")
#   t_l_im_train = tf.data.Dataset.from_tensor_slices(im_l_array_train)
#   t_l_im_test = tf.data.Dataset.from_tensor_slices(im_l_array_test)

#   specfile.write(str(t_l_im_train.element_spec) + '\n')
#   specfile.write(str(t_l_im_test.element_spec) + '\n')

#   print('Saving tensors...')
#   tf.data.experimental.save(t_l_im_train, rootpath + "/Tensors/l_im_train_tensor")
#   tf.data.experimental.save(t_l_im_test, rootpath + "/Tensors/l_im_test_tensor")
#   del t_l_im_train, t_l_im_test

# if frames:
#   print("Saving large image numpy files...")
#   np.save(rootpath + "/DataFrames/im_l_array_train.npy", im_l_array_train)
#   np.save(rootpath + "/DataFrames/im_l_array_test.npy", im_l_array_test)

# del im_l_array_train, im_l_array_test

# ###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# print("loading in small image arrays...")
# ###

# for a in range(num_arrays):
#   list_of_arrays.append(np.load(rootpath + "/Images/image_s_%02d.npy" % a))
# s_image_array = np.concatenate(list_of_arrays)
# list_of_arrays = []

# ###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# print("Split small images for train/test...")
# ###

# im_s_array_train, im_s_array_test, y_train, y_test = train_test_split(
#     s_image_array,
#     y,
#     test_size=0.2,
#     random_state=123456,
#     stratify = y)

# del s_image_array, y

# y_train = keras.utils.to_categorical(y_train, 6)
# y_test = keras.utils.to_categorical(y_test, 6)

# ###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# ###

# if sets:
#   print("Converting small image arrays to tensor files...")
#   t_s_im_train = tf.data.Dataset.from_tensor_slices(im_s_array_train)
#   t_s_im_test = tf.data.Dataset.from_tensor_slices(im_s_array_test)
#   t_y_train = tf.data.Dataset.from_tensor_slices(y_train)
#   t_y_test = tf.data.Dataset.from_tensor_slices(y_test)

#   specfile.write(str(t_s_im_train.element_spec) + '\n')
#   specfile.write(str(t_s_im_test.element_spec) + '\n')
#   specfile.write(str(t_y_train.element_spec) + '\n')
#   specfile.write(str(t_y_test.element_spec) + '\n')

#   print('Saving tensors...')
#   tf.data.experimental.save(t_s_im_train, rootpath + "/Tensors/s_im_train_tensor")
#   tf.data.experimental.save(t_s_im_test, rootpath + "/Tensors/s_im_test_tensor")
#   tf.data.experimental.save(t_y_train, rootpath + "/Tensors/y_train_tensor")
#   tf.data.experimental.save(t_y_test, rootpath + "/Tensors/y_test_tensor")
#   del t_s_im_train, t_s_im_test, t_y_train, t_y_test
  
# if frames:
#   print("Saving small image numpy files...")
#   np.save(rootpath + "/DataFrames/im_s_array_train.npy", im_s_array_train)
#   np.save(rootpath + "/DataFrames/im_s_array_test.npy", im_s_array_test)

#   pd.to_pickle(y_train, rootpath + "/DataFrames/y_train_df.pkl")
#   pd.to_pickle(y_test, rootpath + "/DataFrames/y_test_df.pkl")


# del im_s_array_train, im_s_array_test, y_train, y_test

# ###
# time_elapsed = time.time() - time_start
# time_start = time.time()
# print("elapsed time = " + str(time_elapsed))
# ###

# specfile.close()
# print("END")
