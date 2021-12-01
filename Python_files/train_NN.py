#~~ TRAIN_NN.PY ~~# 
# Will write this later

# Training parameters
batch_size = 2000
no_dense_layers = 3
HL_shape = (20,)
im_l_shape = (21,21,1)
im_s_shape = (11,11,1)
stop_patience = 20
no_epochs = 1
import datetime
model_name = 'Full_model' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

cache_dataset = True
save_model = False
training_parameters = [batch_size, no_dense_layers, stop_patience, no_epochs]

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
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize
import time



# load data
X_train = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/X_train_tensor", element_spec = TensorSpec(shape=(20,), dtype=tf.float64, name=None)).batch(batch_size)
X_test = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/X_test_tensor", element_spec = TensorSpec(shape=(20,), dtype=tf.float64, name=None)).batch(batch_size)
y_train = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/y_train_tensor", element_spec = TensorSpec(shape=(6,), dtype=tf.float32, name=None)).batch(batch_size)
y_test = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/y_test_tensor", element_spec = TensorSpec(shape=(6,), dtype=tf.float32, name=None)).batch(batch_size)

l_im_train = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/l_im_train_tensor", element_spec = TensorSpec(shape=(21,21), dtype=tf.uint8, name=None)).batch(batch_size)
l_im_test = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/l_im_test_tensor", element_spec = TensorSpec(shape=(21,21), dtype=tf.uint8, name=None)).batch(batch_size)
s_im_train = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/s_im_train_tensor", element_spec = TensorSpec(shape=(11,11), dtype=tf.uint8, name=None)).batch(batch_size)
s_im_test = tf.data.experimental.load("/vols/cms/fjo18/Masters2021/Tensors/s_im_test_tensor", element_spec = TensorSpec(shape=(11,11), dtype=tf.uint8, name=None)).batch(batch_size)

train_inputs = tf.data.Dataset.zip((l_im_train, s_im_train, X_train))
test_inputs = tf.data.Dataset.zip((l_im_test, s_im_test, X_test))
# combining three input variables into an object

train_batch = tf.data.Dataset.zip((train_inputs, y_train))
test_batch = tf.data.Dataset.zip((test_inputs, y_test))
# Combining input variables with flags

train_batch = train_batch.prefetch(tf.data.AUTOTUNE)
test_batch = test_batch.prefetch(tf.data.AUTOTUNE)
# prefetching a number of batches, to make loading more efficient possibly

if cache_dataset:
    train_batch = train_batch.cache()
    test_batch = test_batch.cache()
    print('caching dataset')
# caching may well save loading time but not sure yet - for image based the cache does not fit in memory
# may have to partially cache instead of caching entire dataset (i.e. only cache HL variables or something)

# create model
def CNN_creator_3input(inputshape_1, inputshape_2, inputshape_3, convlayers, denselayers, kernelsize = (3,3), learningrate = 0.001):
    # Inputshape should be a 3-comp tuple, where 1st two els are height x width and 3rd is no. layers
    # conv/denselayers denote number of convolutional and dense layers in network
    # convlayers should be a tuple
    # dense necessarily come after convolutional
    poolingsize = (2,2)
    no_conv_layers_l = convlayers[0]
    no_conv_layers_s = convlayers[1]
    no_dense_layers = denselayers
    image_input_l = keras.Input(shape = inputshape_1)
    image_input_s = keras.Input(shape = inputshape_2)
    input_hl = keras.Input(shape = inputshape_3)
    
    x_l = layers.Conv2D(32, kernelsize, activation="relu", padding="same", name = "L_Conv_0")(image_input_l)
    y_l = layers.MaxPooling2D(pool_size=poolingsize, name = "L_Pooling_0")(x_l)
    
    x_s = layers.Conv2D(32, kernelsize, activation="relu", padding="same", name = "S_Conv_0")(image_input_s)
    y_s = layers.MaxPooling2D(pool_size=poolingsize, name = "S_Pooling_0")(x_s)
    
    # For every layer, have (area of kernel + 1) * no_kernels parameters (not sure what the extra one is but maybe the filter weight?)
    for a in range(no_conv_layers_l-1):
        x_l = layers.Conv2D(32 *(a+2), kernelsize, activation="relu", padding="same", name = "L_Conv_" + str(a+1))(y_l)
        y_l = layers.MaxPooling2D(pool_size=poolingsize, name = "L_Pooling_" + str(a+1))(x_l)
        
    for a in range(no_conv_layers_s - 1):
        x_s = layers.Conv2D(32*(a+2), kernelsize, activation="relu", padding="same", name = "S_Conv_" + str(a+1))(y_s)
        y_s = layers.MaxPooling2D(pool_size=poolingsize, name = "S_Pooling_" + str(a+1))(x_s)
    
    x_l = layers.Flatten(name = "L_Flatten")(y_l)
    x_s = layers.Flatten(name = "S_Flatten")(y_s)
    # Flatten output into 1D, so can be applied to dense layers more easily
    x = layers.concatenate([x_l, x_s, input_hl])
    for a in range(no_dense_layers):
        y = layers.Dense(88, activation= "relu", name = "hidden_" + str(a))(x)
        x = layers.BatchNormalization(name = "BatchNorm_" + str(a+1))(y)
    
    outputs = layers.Dense(6, name = "outputs", activation = "softmax")(x)

    model = keras.Model(inputs=[image_input_l, image_input_s, input_hl], outputs=outputs, name="CNN_model_test")
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=learningrate), metrics=["accuracy"],)
    model.summary()
    return model

model = CNN_creator_3input(im_l_shape, im_s_shape, HL_shape, (4,3), no_dense_layers)

early_stop = EarlyStopping(monitor = 'val_loss', patience = stop_patience)
history = History()

# fit model
time_start = time.time()
print("Training model")
print(training_parameters)

model.fit(train_batch,
#           batch_size=200,
          epochs=no_epochs,
          callbacks=[history, early_stop],
          validation_data = test_batch) 

if save_model:
    model.save("/vols/cms/fjo18/Masters2021/Models/%s" % model_name)

time_elapsed = time.time() - time_start 
time_start = time.time()
print("elapsed time = " + str(time_elapsed))

