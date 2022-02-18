## Tensorflow testing area ##
import tensorflow as tf
import pandas as pd
import time
import numpy as np
filepath = "/vols/cms/fjo18/Masters2021/C_DataFrames/DataFrames3_DM2/"
rootpath_save = "/vols/cms/fjo18/Masters2021"
X_test = pd.read_pickle(filepath + "X_test_df.pkl").head(1000000)
l_im_test = np.load(filepath + "im_l_array_test.npy")[:1000000]
s_im_test = np.load(filepath + "im_s_array_test.npy")[:1000000]


featurenames_hl = list(X_test.columns)
npa = X_test.to_numpy()
feature_description = {}
fd_hl = {}
fd_im_l = {}
fd_im_s = {}
for a in range(npa.shape[1]):
    feature_description[featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
    fd_hl[featurenames_hl[a]] = tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
feature_description["large_image"] = tf.io.VarLenFeature(tf.int64)
feature_description["small_image"] = tf.io.VarLenFeature(tf.int64)
fd_im_l["large_image"] = tf.io.VarLenFeature(tf.int64)
fd_im_s["small_image"] = tf.io.VarLenFeature(tf.int64)


print("Finding phis and etas")
time_start = time.time()
filenames = []

dat_len = int(npa.shape[0] * 1)
print(dat_len)

filename_hl = rootpath_save + '/tf_folder/events_hl.tfrecords'
# filename_im_l = rootpath_save + '/tf_folder/events_im_l.tfrecords'
# filename_im_s = rootpath_save + '/tf_folder/events_im_s.tfrecords'
filenames.append(filename_hl)

with tf.io.TFRecordWriter(filename_hl) as writer:
  for a in range(dat_len):
    event_dict = {}
    for b in range(npa.shape[1]):
      event_dict[featurenames_hl[b]] = tf.train.Feature(float_list=tf.train.FloatList(value=[npa[a][b]]))
    event_dict["large_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=l_im_test[a].flatten()))
    event_dict["small_image"] = tf.train.Feature(int64_list=tf.train.Int64List(value=s_im_test[a].flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=event_dict))
    writer.write(example.SerializeToString())

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))


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

def parse_function_tensor_slices(example_proto):
  parsed = tf.io.parse_example(example_proto, fd_im_l)
  return parsed


raw_dataset = tf.data.TFRecordDataset(filenames)

# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))
parsed_dataset_hl = raw_dataset.map(parse_function_hl)
parsed_dataset_im_l = raw_dataset.map(parse_function_im_l)
parsed_dataset_im_s = raw_dataset.map(parse_function_im_s)


# parsed_dataset
for parsed_record in parsed_dataset_im_l.take(5):
  print(repr(parsed_record))
for event in parsed_dataset_im_l.take(1):
  print(event["large_image"])
  event["large_image"] = sparse_remove(event["large_image"])
  print(event["large_image"])
# def resize_l_image(feature):
for element in parsed_dataset_im_s.take(1):
  print(element["small_image"])
for element in parsed_dataset_hl.take(1):
  print(element)

full_dataset = tf.data.Dataset.zip((parsed_dataset_hl, parsed_dataset_im_l, parsed_dataset_im_s))
train_batch = tf.data.Dataset.zip((full_dataset, parsed_dataset_im_l))

for a in train_batch.take(1):
  print(a)