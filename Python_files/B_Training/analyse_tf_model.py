from model_object import parameter_parser, hep_model
rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

default_filepath = "/D_Models/Models3_TF/"
# model_name = "/LSH_model_0.759_20220307_155115"
# model_name = "/LSH_model_0.738_20220308_140330"
model_name = "/LSH_model_0.736_20220308_145600"

# Filenames = [ rootpath_save + '/E_TFRecords/dm%s.tfrecords' % a for a in range(6)]
Filenames_3in = [ rootpath_save + '/E_TFRecords/dm%s_3in.tfrecords' % a for a in range(6)]
#Filenames_3in = Filenames_3in[0]
# Weights = [1.0,1.0,1.0,1.0,1.0,1.0]
Weights = [1.0,1.0,1.0,0.0,0.0,0.0]

#Weights = Weights[0]
jez = hep_model(rootpath_load, rootpath_save, default_filepath, model_name)
jez.prep_for_analysis_tf(Filenames_3in, Weights, True, 50, 1000)
jez.produce_graphs()

