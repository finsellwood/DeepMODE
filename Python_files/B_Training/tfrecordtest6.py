rootpath_save = "/vols/cms/fjo18/Masters2021"
import tensorflow as tf
import numpy as np
Filenames_3in = [ rootpath_save + '/E_TFRecords/dm%s_3in.tfrecords' % a for a in range(6)]

no_features = 29
feature_description = {}
fd_hl = {}
fd_im_l = {}
fd_im_s = {}
fd_flag = {}

feature_description["hl"] =  tf.io.FixedLenFeature([29], tf.float32)
feature_description["large_image"] = tf.io.FixedLenFeature([21,21,7], tf.int64)
feature_description["small_image"] = tf.io.FixedLenFeature([11,11,7], tf.int64)
# feature_description["flag"] = tf.io.FixedLenFeature([6],tf.int64)
fd_im_l["large_image"] = tf.io.FixedLenFeature([21,21,7], tf.int64)
fd_im_s["small_image"] = tf.io.FixedLenFeature([11,11,7], tf.int64)
fd_flag["Outputs"] = tf.io.FixedLenFeature([6],tf.int64)
fd_hl["hl"] =  tf.io.FixedLenFeature([29], tf.float32)


raw_dataset = tf.data.TFRecordDataset([Filenames_3in[2]])

def parse_function_flag_group(example_proto):
    parsed = tf.io.parse_example(example_proto, fd_flag)
    # parsed["Outputs"] = tf.data.Dataset.from_tensor_slices([parsed["Outputs"][0], parsed["Outputs"][1] + parsed["Outputs"][2], \
    #     0, parsed["Outputs"][3], parsed["Outputs"][4], parsed["Outputs"][5]])
    return parsed

flag_dataset = raw_dataset.map(parse_function_flag_group)
new_flag_dataset = flag_dataset.map(lambda a:{"Outputs": [a["Outputs"][0], a["Outputs"][1] + a["Outputs"][2]\
    , 0, a["Outputs"][3], a["Outputs"][4], a["Outputs"][5]] })

def decode_fn(ds):
    ds["Outputs"] = [ds["Outputs"][0], ds["Outputs"][1] + ds["Outputs"][2]\
    , 0, ds["Outputs"][3], ds["Outputs"][4], ds["Outputs"][5]]
    return ds
test_flag_dataset = flag_dataset.apply(decode_fn)

for event in new_flag_dataset.take(10):
    print(event["Outputs"])
    
for event in flag_dataset.take(10):
    print(event["Outputs"])