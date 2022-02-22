## Tensorflow testing area ##
import tensorflow as tf
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
filepath = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2/"
rootpath_save = "/vols/cms/fjo18/Masters2021"
featurenames_hl = list(['pi_E_2', 'pi2_E_2', 'pi3_E_2', 'pi0_E_2', 'n_gammas_2', 'sc1_r9_5x5_2',
       'sc1_ietaieta_5x5_2', 'sc1_Nclusters_2', 'tau_E_2', 'tau_decay_mode_2',
       'pt_2', 'pi0_2mass', 'rho_mass', 'E_gam/E_tau', 'E_pi/E_tau',
       'E_pi0/E_tau', 'tau_eta', 'delR_gam', 'delPhi_gam', 'delEta_gam',
       'delR_xE_gam', 'delPhi_xE_gam', 'delEta_xE_gam', 'delR_pi', 'delPhi_pi',
       'delEta_pi', 'delR_xE_pi', 'delPhi_xE_pi', 'delEta_xE_pi'])

no_features = len(featurenames_hl)
feature_description = {}
fd_hl = {}
fd_im_l = {}
fd_im_s = {}
fd_flag = {}
for a in range(no_features):
    feature_description[featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
    fd_hl[featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
feature_description["large_image"] = tf.io.VarLenFeature(tf.int64)
feature_description["small_image"] = tf.io.VarLenFeature(tf.int64)
fd_im_l["large_image"] = tf.io.VarLenFeature(tf.int64)
fd_im_s["small_image"] = tf.io.VarLenFeature(tf.int64)
fd_flag["flag"] = tf.io.FixedLenFeature([],tf.int16)


print("Finding phis and etas")
time_start = time.time()
filenames = []

filenames = [ rootpath_save + '/E_TFRecords/dm%s.tfrecords' % a for a in range(6)]
# filename_im_l = rootpath_save + '/tf_folder/events_im_l.tfrecords'
# filename_im_s = rootpath_save + '/tf_folder/events_im_s.tfrecords'
# filenames.append(filename_hl)


### Now retrieve from files ###
def sparse_remove(sparse_tensor):
  return tf.sparse.retain(sparse_tensor, tf.not_equal(sparse_tensor.values, 0))
def parse_function_hl(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, fd_hl)
  return parsed
def parse_function_im_l(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, fd_im_l)
  # parsed["large_image"] = tf.sparse.reshape(parsed["large_image"], shape=(21,21,6))
  parsed["large_image"] = tf.sparse.reshape(sparse_remove(parsed["large_image"]), shape=(21,21,7))
  return parsed
def parse_function_im_s(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, fd_im_s)
  # parsed["large_image"] = tf.sparse.reshape(parsed["large_image"], shape=(21,21,6))
  parsed["small_image"] = tf.sparse.reshape(sparse_remove(parsed["small_image"]), shape=(11,11,7))
  return parsed


raw_datasets = []
for a in range(len(filenames)):
  raw_datasets.append(tf.data.TFRecordDataset([filenames[a]]))

# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))
hl_datasets = [a.map(parse_function_hl) for a in raw_datasets]
im_l_datasets = [a.map(parse_function_im_l) for a in raw_datasets]
im_s_datasets = [a.map(parse_function_im_s) for a in raw_datasets]

# parsed_dataset_hl = raw_dataset.map(parse_function_hl)
# parsed_dataset_im_l = raw_dataset.map(parse_function_im_l)
# parsed_dataset_im_s = raw_dataset.map(parse_function_im_s)


# parsed_dataset
for parsed_record in parsed_dataset_im_l.take(5):
  print(repr(parsed_record))

# def resize_l_image(feature):
for element in parsed_dataset_im_s.take(1):
  print(element["small_image"])
for element in parsed_dataset_hl.take(1):
  print(element)

weights1 = [1.0,1.0,1.0,1.0,1.0,1.0]
weights2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
sample_dataset = tf.data.Dataset.sample_from_datasets(
    hl_datasets, weights=weights2)

full_dataset = tf.data.Dataset.zip((parsed_dataset_hl, parsed_dataset_im_l, parsed_dataset_im_s))
train_batch = tf.data.Dataset.zip((full_dataset, parsed_dataset_im_l))


for a in train_batch.take(1):
  print(a)

data = []
for element in sample_dataset.take(1000):
  data.append(element["tau_decay_mode_2"].numpy())

plt.hist(data)
plt.savefig("histogram.png")