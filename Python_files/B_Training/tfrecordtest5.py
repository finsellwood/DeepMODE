## Tensorflow testing area ##
from re import I
import tensorflow as tf
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import json
filepath = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2/"
rootpath_save = "/vols/cms/fjo18/Masters2021"
featurenames_hl = list(['pi_E_2', 'pi2_E_2', 'pi3_E_2', 'pi0_E_2', 'n_gammas_2', 'sc1_r9_5x5_2',
       'sc1_ietaieta_5x5_2', 'sc1_Nclusters_2', 'tau_E_2', 'tau_decay_mode_2',
       'pt_2', 'pi0_2mass', 'rho_mass', 'E_gam/E_tau', 'E_pi/E_tau',
       'E_pi0/E_tau', 'tau_eta', 'delR_gam', 'delPhi_gam', 'delEta_gam',
       'delR_xE_gam', 'delPhi_xE_gam', 'delEta_xE_gam', 'delR_pi', 'delPhi_pi',
       'delEta_pi', 'delR_xE_pi', 'delPhi_xE_pi', 'delEta_xE_pi'])
filepath = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2/"
rootpath_save = "/vols/cms/fjo18/Masters2021"
X_test = pd.read_pickle(filepath + "X_test_df.pkl").head(10000)
l_im_test = np.load(filepath + "im_l_array_test.npy")[:10000]
s_im_test = np.load(filepath + "im_s_array_test.npy")[:10000]
npa = X_test.to_numpy()


no_features = len(featurenames_hl)
feature_description = {}
fd_hl = {}
fd_im_l = {}
fd_im_s = {}
fd_flag = {}
no_hl_features = npa.shape[-1]

feature_description["hl"] =  tf.io.FixedLenFeature([no_hl_features], tf.float32)
feature_description["large_image"] = tf.io.FixedLenFeature([21,21,7], tf.int64)
feature_description["small_image"] = tf.io.FixedLenFeature([11,11,7], tf.int64)
# feature_description["flag"] = tf.io.FixedLenFeature([6],tf.int64)
fd_im_l["large_image"] = tf.io.FixedLenFeature([21,21,7], tf.int64)
fd_im_s["small_image"] = tf.io.FixedLenFeature([11,11,7], tf.int64)
fd_flag["flag"] = tf.io.FixedLenFeature([6],tf.int64)
fd_hl["hl"] =  tf.io.FixedLenFeature([no_hl_features], tf.float32)

# fd_im_l["flag"] = tf.io.FixedLenFeature([],tf.int64)m


print("Finding phis and etas")
time_start = time.time()
filenames = []

filenames = [ rootpath_save + '/E_TFRecords/dm%s_new.tfrecords' % a for a in range(6)]
filename_hl = filenames[0]
modeflag = 1
onehot_flag = [0,0,0,0,0,0]
onehot_flag[modeflag] = 1
with tf.io.TFRecordWriter(filename_hl) as writer:
  for a in range(1):
    event_dict = {}
    event_dict["hl"] = tf.train.Feature(float_list=tf.train.FloatList(value=npa[a].flatten()))
    event_dict["large_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=l_im_test[a].flatten()))
    event_dict["small_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=s_im_test[a].flatten()))
    event_dict["flag"] = tf.train.Feature(int64_list=tf.train.Int64List(value=onehot_flag))
    example = tf.train.Example(features=tf.train.Features(feature=event_dict))
    # print(example)
    print(event_dict["hl"])
    writer.write(example.SerializeToString())



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
def parse_function_flag(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, fd_flag)
  #parsed["flag"] = tf.sparse.reshape(parsed["flag"], shape=(6,))
  return parsed
def parse_function_im_l(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, fd_im_l)
  # parsed["large_image"] = tf.sparse.reshape(parsed["large_image"], shape=(21,21,6))
  #parsed["large_image"] = tf.sparse.reshape(sparse_remove(parsed["large_image"]), shape=(21,21,7))
  return parsed
def parse_function_im_s(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, fd_im_s)
  # parsed["large_image"] = tf.sparse.reshape(parsed["large_image"], shape=(21,21,6))
  parsed["small_image"] = tf.sparse.reshape(sparse_remove(parsed["small_image"]), shape=(11,11,7))
  return parsed

def parse_function_all(example_proto):
  parsed = tf.io.parse_example(example_proto, feature_description)
  #parsed["large_image"] = tf.sparse.reshape(sparse_remove(parsed["large_image"]), shape=(21,21,7))
  #parsed["small_image"] = tf.sparse.reshape(sparse_remove(parsed["small_image"]), shape=(11,11,7))
  return parsed



filenames = [filename_hl]
raw_datasets = []
for a in range(len(filenames)):
  raw_datasets.append(tf.data.TFRecordDataset([filenames[a]]))

# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))
hl_datasets = [a.map(parse_function_hl) for a in raw_datasets]
flag_datasets = [a.map(parse_function_flag) for a in raw_datasets]
im_l_datasets = [a.map(parse_function_im_l) for a in raw_datasets]
im_s_datasets = [a.map(parse_function_im_s) for a in raw_datasets]

full_datasets = [a.map(parse_function_all) for a in raw_datasets]

# parsed_dataset_hl = raw_dataset.map(parse_function_hl)
# parsed_dataset_im_l = raw_dataset.map(parse_function_im_l)
# parsed_dataset_im_s = raw_dataset.map(parse_function_im_s)


# # parsed_dataset
# for parsed_record in parsed_dataset_im_l.take(5):
#   print(repr(parsed_record))

# # def resize_l_image(feature):
# for element in parsed_dataset_im_s.take(1):
#   print(element["small_image"])
# for element in parsed_dataset_hl.take(1):
#   print(element)

weights1 = [1.0,1.0,1.0,1.0,1.0,1.0]
weights2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
weights3 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
weights3 = [1.0]
sample_dataset_hl = tf.data.Dataset.sample_from_datasets(
    hl_datasets, weights=weights3, seed=1234, stop_on_empty_dataset=True)
sample_dataset_im_l = tf.data.Dataset.sample_from_datasets(
    im_l_datasets, weights=weights3, seed=1234, stop_on_empty_dataset=True)
sample_dataset_im_l = tf.data.Dataset.sample_from_datasets(
    im_l_datasets, weights=weights3, seed=1234, stop_on_empty_dataset=True)
sample_dataset_flag = tf.data.Dataset.sample_from_datasets(
    flag_datasets, weights=weights3, seed=1234, stop_on_empty_dataset=True)
# full_dataset = tf.data.Dataset.zip((parsed_dataset_hl, parsed_dataset_im_l, parsed_dataset_im_s))
# train_batch = tf.data.Dataset.zip((full_dataset, parsed_dataset_im_l))


# for a in sample_dataset.take(1):
#   print(a)

data = []
for element in sample_dataset_flag.take(10000):
  data.append(element["flag"].numpy())
  # print(element["flag"].numpy())
# for element in sample_dataset_im_l.take(10):
#   # data.append(element["flag"].numpy())
#   print(element["flag"].numpy())
plt.hist(data)
plt.savefig("histogram.png")
plt.show()
plt.clf()



for element in full_datasets[0].take(1):
  print(element)


for element in im_l_datasets[0].take(1):
  print(element)

##%%

from model_object import parameter_parser, hep_model
rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

default_filepath = "/D_Models/Models3_TF/"
model_name = "model"
mez_filepath = "/vols/cms/fjo18/Masters2021/D_Models/Models3_DM2/LSH_model_0.715_20220205_195903"


# Filenames = [ rootpath_save + '/E_TFRecords/dm%s.tfrecords' % a for a in range(6)]
Filenames_3in = [ rootpath_save + '/E_TFRecords/dm%s_3in.tfrecords' % a for a in range(6)]
#Filenames_3in = Filenames_3in[0]
Weights = [1.0,1.0,1.0,0.0,0.0,0.0]
#Weights = Weights[0]
jez = hep_model(rootpath_load, rootpath_save, default_filepath, model_name)
jez.no_epochs = 1
jez.create_featuredesc()
jez.batch_size = 64
jez.load_tfrecords(Filenames_3in, Weights, True, new_flags = True)
jez.load_model()

# jez.flag_datasets = [a.map(lambda a:{"Outputs": [a["Outputs"][0], a["Outputs"][1] + a["Outputs"][2], 0, a["Outputs"][3], a["Outputs"][4], a["Outputs"][5]] }) for a in jez.flag_datasets]