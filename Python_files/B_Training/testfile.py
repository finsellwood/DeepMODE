import json
paramdict = {"batch_size": 1000, "conv_layers": [[4, 4], [4, 3]], "dense_layers": [[6, 32, False], [4, 512, True]], "inc_dropout": True, "dropout_rate": [0.1, 0.3], "use_inputs": [True, True, True], "learningrate": 0.001, "no_epochs": 25, "stop_patience": 25, "use_res_blocks": False, "drop_variables": False, "flat_preprocess": True, "HL_shape": [29], "im_l_shape": [21, 21, 7], "im_s_shape": [11, 11, 7], "no_modes": 3, "data_folder": "/C_DataFrames/DataFrames3_DM2/", "model_folder": "/D_Models/Models3_DM2/", "save_model": True, "small_dataset": False, "small_dataset_size": 10000}
param_file = open("_params.json", 'w')
json.dump(paramdict, param_file)
param_file.close()