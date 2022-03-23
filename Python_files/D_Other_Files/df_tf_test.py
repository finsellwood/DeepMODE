#/DataFrames3_DM2_no_pi0 <-- small data location

import sys
sys.path.append("/home/hep/fjo18/CMSSW_10_2_19/src/UserCode/DeepLearning/Python_files/B_Training/")

from model_object import hep_model

vols_path = "/vols/cms/fjo18/Masters2021/"
model_folder = "/D_Models/Models3_TF/"
model_name = "LSH_model_0.722_20220323_104204"
data_path = "C_DataFrames/DataFrames3_DM/"

model_object = hep_model(vols_path, vols_path, model_folder, model_name)
model_object.data_folder = data_path

model_object.load_data()
model_object.load_model()

model_object.predict_results()

response = model_object.ypred()

print(response)