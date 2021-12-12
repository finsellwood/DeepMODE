#~~ TRAIN_NN.PY ~~# 
# Will write this later
rootpath = "/vols/cms/fjo18/Masters2021"
# Training parameters
batch_size = 1024
#dense_layers = [(4,128, False), (2, 54, False)]
dense_layers = [(2,40, True), (2, 40, True)]
HL_shape = (21,)
im_l_shape = (21,21,1)
im_s_shape = (11,11,1)
stop_patience = 20
no_epochs = 50
learningrate = 0.01
import datetime
from math import ceil
model_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# cache_dataset = True
view_model = False
# DOESN'T WORK - HAVE TO INSTALL IF NECESSARY
save_model = True
small_dataset = False
training_parameters = [batch_size, dense_layers, learningrate, stop_patience, no_epochs]

# Load packages
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
# Code will now print the device on which it is running
from tensorflow import TensorSpec
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Normalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize, plot_model

import time
# #fix#
# import pickle5 as pickle


# load data

X_train = pd.read_pickle(rootpath + "/DataFrames/X_train_df.pkl")
X_test = pd.read_pickle(rootpath + "/DataFrames/X_test_df.pkl")
y_train = pd.read_pickle(rootpath + "/DataFrames/y_train_df.pkl")
y_test = pd.read_pickle(rootpath + "/DataFrames/y_test_df.pkl")
l_im_train = np.load(rootpath + "/DataFrames/im_l_array_train.npy")
l_im_test = np.load(rootpath + "/DataFrames/im_l_array_test.npy")
s_im_train = np.load(rootpath + "/DataFrames/im_s_array_train.npy")
s_im_test = np.load(rootpath + "/DataFrames/im_s_array_test.npy")

if small_dataset:
    X_train = X_train.head(800000)
    X_test = X_test.head(200000)
    y_train = y_train[:800000]
    y_test = y_test[:200000]
    l_im_train = l_im_train[:800000]
    l_im_test = l_im_test[:200000]
    s_im_train = s_im_train[:800000]
    s_im_test = s_im_test[:200000]



train_inputs = [l_im_train, s_im_train, X_train]
test_inputs = [l_im_test, s_im_test, X_test]

# X_train = tf.data.experimental.load(rootpath + "/Tensors/X_train_tensor", element_spec = TensorSpec(shape=(20,), dtype=tf.float64, name=None)).batch(batch_size)
# X_test = tf.data.experimental.load(rootpath + "/Tensors/X_test_tensor", element_spec = TensorSpec(shape=(20,), dtype=tf.float64, name=None)).batch(batch_size)
# y_train = tf.data.experimental.load(rootpath + "/Tensors/y_train_tensor", element_spec = TensorSpec(shape=(6,), dtype=tf.float32, name=None)).batch(batch_size)
# y_test = tf.data.experimental.load(rootpath + "/Tensors/y_test_tensor", element_spec = TensorSpec(shape=(6,), dtype=tf.float32, name=None)).batch(batch_size)

# l_im_train = tf.data.experimental.load(rootpath + "/Tensors/l_im_train_tensor", element_spec = TensorSpec(shape=(21,21), dtype=tf.uint8, name=None)).batch(batch_size)
# l_im_test = tf.data.experimental.load(rootpath + "/Tensors/l_im_test_tensor", element_spec = TensorSpec(shape=(21,21), dtype=tf.uint8, name=None)).batch(batch_size)
# s_im_train = tf.data.experimental.load(rootpath + "/Tensors/s_im_train_tensor", element_spec = TensorSpec(shape=(11,11), dtype=tf.uint8, name=None)).batch(batch_size)
# s_im_test = tf.data.experimental.load(rootpath + "/Tensors/s_im_test_tensor", element_spec = TensorSpec(shape=(11,11), dtype=tf.uint8, name=None)).batch(batch_size)

# train_inputs = tf.data.Dataset.zip((l_im_train, s_im_train, X_train))
# test_inputs = tf.data.Dataset.zip((l_im_test, s_im_test, X_test))
# # combining three input variables into an object

# train_batch = tf.data.Dataset.zip((train_inputs, y_train))
# test_batch = tf.data.Dataset.zip((test_inputs, y_test))
# # Combining input variables with flags

# train_batch = train_batch.prefetch(tf.data.AUTOTUNE)
# test_batch = test_batch.prefetch(tf.data.AUTOTUNE)
# # prefetching a number of batches, to make loading more efficient possibly

# if cache_dataset:
#     train_batch = train_batch.cache()
#     test_batch = test_batch.cache()
#     print('caching dataset')
# # caching may well save loading time but not sure yet - for image based the cache does not fit in memory
# # may have to partially cache instead of caching entire dataset (i.e. only cache HL variables or something)

# create model
def CNN_creator_3input(inputshape_l, inputshape_s, inputshape_hl, convlayers, denselayers, kernelsize = (3,3), learningrate = 0.001, model_image = False):
    # Inputshape should be a 3-comp tuple, where 1st two els are height x width and 3rd is no. layers
    # conv/denselayers denote number of convolutional and dense layers in network
    # convlayers should be a tuple
    # dense layers is 2x2 matrix, first val for dense layers for only HL, second is for the final dense layers
    # - second val in each tuple is the width of the dense layer
    # dense necessarily come after convolutional
    poolingsize = (2,2)
    no_conv_layers_l = convlayers[0]
    no_conv_layers_s = convlayers[1]

    no_dense_hl = denselayers[0][0]
    width_dense_hl = denselayers[0][1]
    decrease_dense_hl = denselayers[0][2]
    no_dense_full = denselayers[1][0]
    width_dense_full = denselayers[1][1]
    decrease_dense_full = denselayers[1][2]

    image_input_l = keras.Input(shape = inputshape_l)
    image_input_s = keras.Input(shape = inputshape_s)
    input_hl = keras.Input(shape = inputshape_hl)
    norm_hl = keras.layers.Normalization(name = 'Input_Normalisation')(input_hl)
    # Normalise the hl inputs (feature wise) before running them
    
    x_l = layers.Conv2D(32, kernelsize, activation="relu", padding="same", name = "L_Conv_0")(image_input_l)
    y_l = layers.MaxPooling2D(pool_size=poolingsize, name = "L_Pooling_0")(x_l)
    
    x_s = layers.Conv2D(32, kernelsize, activation="relu", padding="same", name = "S_Conv_0")(image_input_s)
    y_s = layers.MaxPooling2D(pool_size=poolingsize, name = "S_Pooling_0")(x_s)

    x_hl = layers.Dense(width_dense_hl, activation= "relu", name = "hidden_HL_0")(norm_hl)
    y_hl = layers.BatchNormalization(name = "BatchNorm_HL_0")(x_hl)

    # For every layer, have (area of kernel + 1) * no_kernels parameters (not sure what the extra one is but maybe the filter weight?)
    for a in range(no_conv_layers_l-1):
        do_l = layers.Dropout(0.4)(y_l)
        x_l = layers.Conv2D(32 *(a+2), kernelsize, activation="relu", padding="same", name = "L_Conv_" + str(a+1))(do_l)
        y_l = layers.MaxPooling2D(pool_size=poolingsize, name = "L_Pooling_" + str(a+1))(x_l)
        
        
    for a in range(no_conv_layers_s - 1):
        do_s = layers.Dropout(0.4)(y_s)
        x_s = layers.Conv2D(32*(a+2), kernelsize, activation="relu", padding="same", name = "S_Conv_" + str(a+1))(do_s)
        y_s = layers.MaxPooling2D(pool_size=poolingsize, name = "S_Pooling_" + str(a+1))(x_s)

    for a in range(no_dense_hl - 1):
        do_hl = layers.Dropout(0.4)(y_hl)
        # Added dropout layers into dense (09.12.21)
        if decrease_dense_hl:
            x_hl = layers.Dense(ceil(width_dense_hl * 0.5 **(a+1)), activation= "relu", name = "hidden_HL_" + str(a+1))(do_hl)
            # Layers get smaller and smaller
        else:
            x_hl = layers.Dense(width_dense_hl, activation= "relu", name = "hidden_HL_" + str(a+1))(do_hl)
            # Layers stay same size

        y_hl = layers.BatchNormalization(name = "BatchNorm_HL_" + str(a+1))(x_hl)
    
    x_l = layers.Flatten(name = "L_Flatten")(y_l)
    x_s = layers.Flatten(name = "S_Flatten")(y_s)
    # Flatten output into 1D, so can be applied to dense layers more easily
    x = layers.concatenate([x_l, x_s, y_hl])

    for a in range(no_dense_full):
        if decrease_dense_full:
            y = layers.Dense(ceil(width_dense_full * 0.5 **a), activation= "relu", name = "hidden_" + str(a))(x)
        else:
            y = layers.Dense(width_dense_full, activation= "relu", name = "hidden_" + str(a))(x)

        x = layers.BatchNormalization(name = "BatchNorm_" + str(a))(y)
    
    outputs = layers.Dense(6, name = "outputs", activation = "softmax")(x)

    model = keras.Model(inputs=[image_input_l, image_input_s, input_hl], outputs=outputs)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=learningrate), metrics=["accuracy"],)
    model.summary()
    if model_image:
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

model = CNN_creator_3input(im_l_shape, im_s_shape, HL_shape, (4,3), dense_layers, learningrate = learningrate, model_image=view_model)

early_stop = EarlyStopping(monitor = 'val_loss', patience = stop_patience)
history = History()

# fit model
time_start = time.time()
print("Training model")
print(training_parameters)

model.fit(train_inputs, y_train,
          batch_size = batch_size,
          epochs = no_epochs,
          callbacks=[history, early_stop],
          validation_data = (test_inputs, y_test)) 


prediction = model.predict(test_inputs)
idx = prediction.argmax(axis=1)
y_pred = (idx[:,None] == np.arange(prediction.shape[1])).astype(float)
flatpred = np.argmax(y_pred, axis=-1)
flattest = np.argmax(y_test, axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

if save_model:
    model.save(rootpath + "/Models/Full_model_%.3f_%s" % (accuracy, model_datetime))

time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))

