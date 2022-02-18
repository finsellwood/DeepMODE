## Tensorflow testing area ##
import tensorflow as tf
import pandas as pd
import time
import numpy as np
filepath = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2/"
rootpath_save = "/vols/cms/fjo18/Masters2021"
X_test = pd.read_pickle(filepath + "X_test_df.pkl").head(1000)
l_im_test = np.load(filepath + "im_l_array_test.npy")[:1000]
s_im_test = np.load(filepath + "im_s_array_test.npy")[:1000]


featurenames_hl = list(X_test.columns)
npa = X_test.to_numpy()
feature_description = {}
feature_description_im_l = { "large_image": tf.io.VarLenFeature(tf.int64)}
feature_description_im_s = { "small_image": tf.io.VarLenFeature(tf.int64)}

for a in range(npa.shape[1]):
    feature_description[featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)


print("Finding phis and etas")
time_start = time.time()
filenames_hl = []
filenames_im_l = []
filenames_im_s = []
dat_len = int(npa.shape[0] * 1)
print(dat_len)

filename_hl = rootpath_save + '/tf_folder/events_hl.tfrecords'
filename_im_l = rootpath_save + '/tf_folder/events_im_l.tfrecords'
filename_im_s = rootpath_save + '/tf_folder/events_im_s.tfrecords'

with tf.io.TFRecordWriter(filename_hl) as writer:
  for a in range(dat_len):
    event_dict = {}
    for b in range(npa.shape[1]):
      event_dict[featurenames_hl[b]] = tf.train.Feature(float_list=tf.train.FloatList(value=[npa[a][b]]))
    example = tf.train.Example(features=tf.train.Features(feature=event_dict))
    writer.write(example.SerializeToString())

with tf.io.TFRecordWriter(filename_im_l) as writer:
  for a in range(dat_len):
    event_dict = {"large_image": tf.train.Feature(int64_list=tf.train.Int64List(value=l_im_test[a].flatten()))}
    example = tf.train.Example(features=tf.train.Features(feature=event_dict))
    writer.write(example.SerializeToString())

with tf.io.TFRecordWriter(filename_im_s) as writer:
  for a in range(dat_len):
    event_dict = {"small_image": tf.train.Feature(int64_list=tf.train.Int64List(value=s_im_test[a].flatten()))}
    example = tf.train.Example(features=tf.train.Features(feature=event_dict))
    writer.write(example.SerializeToString())

filenames_hl.append(filename_hl)
filenames_im_l.append(filename_im_l)
filenames_im_s.append(filename_im_s)

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))

for a in range(dat_len):#npa.shape[0]
  event_dict = {}
  for b in range(npa.shape[1]):
      event_dict[featurenames_hl[b]] = tf.train.Feature(float_list=tf.train.FloatList(value=[npa[a][b]]))
  event_dict_im_l = {"large_image": tf.train.Feature(int64_list=tf.train.Int64List(value=l_im_test[a].flatten()))}
  event_dict_im_s = {"small_image": tf.train.Feature(int64_list=tf.train.Int64List(value=s_im_test[a].flatten()))}

  example = tf.train.Example(features=tf.train.Features(feature=event_dict))
  example_im_l = tf.train.Example(features=tf.train.Features(feature=event_dict_im_l))
  example_im_s = tf.train.Example(features=tf.train.Features(feature=event_dict_im_s))

  filename_hl = rootpath_save + '/tf_folder/event%s_hl.tfrecords' % a
  filename_im_l = rootpath_save + '/tf_folder/event%s_im_l.tfrecords' % a
  filename_im_s = rootpath_save + '/tf_folder/event%s_im_s.tfrecords' % a

  with tf.io.TFRecordWriter(filename_hl) as writer:
    writer.write(example.SerializeToString())
  with tf.io.TFRecordWriter(filename_im_l) as writer:
    writer.write(example_im_l.SerializeToString())
  with tf.io.TFRecordWriter(filename_im_s) as writer:
    writer.write(example_im_s.SerializeToString())
  filenames_hl.append(filename_hl)
  filenames_im_l.append(filename_im_l)
  filenames_im_s.append(filename_im_s)


time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))


### Now retrieve from files ###
def sparse_remove(sparse_tensor):
  return tf.sparse.retain(sparse_tensor, tf.not_equal(sparse_tensor.values, 0))
def parse_function_hl(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, feature_description)
  otherfeature = {"newfeature": parsed["pi0_E_2"]}
  return parsed, otherfeature
def parse_function_im_l(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, feature_description_im_l)
  # parsed["large_image"] = tf.sparse.reshape(parsed["large_image"], shape=(21,21,6))
  parsed["large_image"] = tf.sparse.reshape(sparse_remove(parsed["large_image"]), shape=(21,21,7))
  return parsed
def parse_function_im_s(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  parsed = tf.io.parse_example(example_proto, feature_description_im_s)
  # parsed["large_image"] = tf.sparse.reshape(parsed["large_image"], shape=(21,21,6))
  parsed["small_image"] = tf.sparse.reshape(sparse_remove(parsed["small_image"]), shape=(11,11,7))
  return parsed

raw_dataset_hl = tf.data.TFRecordDataset(filenames_hl)
raw_dataset_im_l = tf.data.TFRecordDataset(filenames_im_l)
raw_dataset_im_s = tf.data.TFRecordDataset(filenames_im_s)
# raw_dataset
# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))
full_dataset = raw_dataset_hl.map(parse_function_hl)
parsed_dataset_hl = full_dataset[0]
parsed_dataset_im_l = raw_dataset_im_l.map(parse_function_im_l)
parsed_dataset_im_s = raw_dataset_im_s.map(parse_function_im_s)


# parsed_dataset
for parsed_record in parsed_dataset_im_l.take(5):
  print(repr(parsed_record))
for event in parsed_dataset_im_l.take(1):
  print(event["large_image"])
  event["large_image"] = sparse_remove(event["large_image"])
  print(event["large_image"])
# def resize_l_image(feature):
for element in parsed_dataset_im_s.take(1):
  print(element)#["small_image"])
for element in parsed_dataset_hl.take(1):
  print(element[1])
