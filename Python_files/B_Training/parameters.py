#~~ parameters.py ~~#
# # File for holding certain parameter presets for models

drop_variables = False
# Training parameters
batch_size = 1000 #1024
stop_patience = 25
no_epochs = 25
learningrate = 0.001
no_modes = 3
data_folder = "/C_DataFrames/DataFrames3_DM2_no_pi0/"
model_folder = "/D_Models/Models3_DM2_no_pi0/"
# Model architecture parameters
#dense_layers = [(4,128, False), (2, 54, False)]
# dense_layers = [(6, 512, True), (4, 32, False)]
dense_layers = [(6, 32, False), (4, 512, True)]
conv_layers = [(4,4), (4,3)]
flat_preprocess = True
# Determines if the initial (no pooling) conv layers have a constant no. filters
# (True = constant)
HL_shape = (26,)
im_l_shape = (21,21,6)
im_s_shape = (11,11,6)
inc_dropout = True
dropout_rate = [0.1, 0.3]
# 1st no. is conv and 2nd is dense
# Convolutional layers should have a much lower dropout rate than dense
use_inputs = [True, True, True]
# A mask to check which inputs to use for the model - above indicates HL only
# use_unnormalised = True
use_res_blocks = False
# Currently not particularly founded use of residual layers 
# Should be applied between convolutional layers, not dense
save_model = True
small_dataset = False
small_dataset_size = 10000

paramdict = {"batch_size":batch_size, "conv_layers":conv_layers, "dense_layers":dense_layers,\
     "inc_dropout":inc_dropout,"dropout_rate": dropout_rate, "use_inputs": use_inputs,\
     "learningrate":learningrate, "no_epochs":no_epochs, "stop_patience":stop_patience,\
     "use_res_blocks":use_res_blocks, "drop_variables":drop_variables, \
     "flat_preprocess":flat_preprocess, "HL_shape":HL_shape, "im_l_shape":im_l_shape,\
     "im_s_shape":im_s_shape, "no_modes":no_modes, "data_folder":data_folder, \
     "model_folder":model_folder, "save_model":save_model, "small_dataset":small_dataset,\
     "small_dataset_size":small_dataset_size,}


# training_parameters = [batch_size, conv_layers, dense_layers, inc_dropout, \
#     dropout_rate, use_inputs, learningrate, no_epochs, stop_patience, save_model, \
#         small_dataset, use_res_blocks, drop_variables, flat_preprocess, HL_shape, im_l_shape, im_s_shape, no_modes]

# training_parameter_names = ["batch size", "conv layers", "dense layers", "include dropout?", \
#     "dropout rate", "inputs mask", "learning rate", "no. epochs", "stop patience", "save model?", \
#     "small dataset?", "Use residual blocks?", "drop some variables?", "use constant filter size?",\
#     "HL Shape?", "Large image shape?", "Small image shape?", "How many decay modes?"]
# For future reference