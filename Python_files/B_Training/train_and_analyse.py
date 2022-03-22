from tkinter.messagebox import NO
from model_object import parameter_parser, hep_model
rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

default_filepath = "/D_Models/Models3_TF/"
model_name = "model"

No_batches_analysis = 100
Batch_size_analysis = 1000

Filenames_3in = [ rootpath_save + '/E_TFRecords/dm%s_3in.tfrecords' % a for a in range(6)]
#Filenames_3in = Filenames_3in[0]
# Weights = [1.0,1.0,1.0,1.0,1.0,0.0]
# Weights = [1.0,1.0,2.0,0.0,0.0,0.0]
Weights = [1.0,1.0,1.0,0.0,0.0,0.0]

# Weights = [0.0,0.0,0.0,1.0,1.0,0.05]
# Weights = [1.0,1.0,0.0,0.0,0.0,0.0]

jez = hep_model(rootpath_load, rootpath_save, default_filepath, model_name)
jez.no_epochs = 25

### TRAIN MODEL ###

jez.do_your_thing_tf(Filenames_3in, Weights, use_as_mask=False, new_flags = False)

### ANALYSE MODEL ###
jez.no_batches = No_batches_analysis
jez.batch_size = Batch_size_analysis
jez.use_datasets = True
jez.create_featuredesc()
jez.load_tfrecords(Filenames_3in, Weights, use_as_mask = True, new_flags=False)
jez.analyse_model_tf(No_batches_analysis)
jez.no_modes = 3
jez.produce_graphs(False)

