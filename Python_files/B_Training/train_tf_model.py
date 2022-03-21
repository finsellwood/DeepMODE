from model_object import parameter_parser, hep_model
rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

default_filepath = "/D_Models/Models3_TF/"
model_name = "model"


# Filenames = [ rootpath_save + '/E_TFRecords/dm%s.tfrecords' % a for a in range(6)]
Filenames_3in = [ rootpath_save + '/E_TFRecords/dm%s_3in.tfrecords' % a for a in range(6)]
#Filenames_3in = Filenames_3in[0]
# Weights = [1.0,1.0,1.0,1.0,1.0,0.0]
# Weights = [1.0,1.0,2.0,0.0,0.0,0.0]
Weights = [1.0,1.0,1.0,0.0,0.0,0.0]

# Weights = [0.0,0.0,0.0,1.0,1.0,0.05]
# Weights = [1.0,1.0,0.0,0.0,0.0,0.0]


#Weights = Weights[0]
jez = hep_model(rootpath_load, rootpath_save, default_filepath, model_name)
jez.no_epochs = 10

jez.do_your_thing_tf(Filenames_3in, Weights, False, new_flags = True)
# jez.doublecheck_tf(Filenames_3in, Weights, False)
#/vols/cms/fjo18/Masters2021/D_Models/Models3_TF/LSH_model_0.715_20220303_152850

# model_name = "LSH_model_0.715_20220303_152850"

# Weights = [1.0,0.0,0.0,0.0,0.0,0.0]
