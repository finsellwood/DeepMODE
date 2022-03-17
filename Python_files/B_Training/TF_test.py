from model_object import *
rootpath_load = "/vols/cms/fjo18/Masters2021"
rootpath_save = "/vols/cms/fjo18/Masters2021"

default_filepath = "/D_Models/Models3_TF/"
# model_name = "/LSH_model_0.759_20220307_155115"
# model_name = "/LSH_model_0.738_20220308_140330"
model_name = "/LSH_model_0.736_20220308_145600"
# model_name = "/LSH_model_0.796_20220315_150659" #3pr model only
# model_name = "/LSH_model_0.852_20220315_160854" #3pr WITHOUT other model
# model_name = "/LSH_model_0.791_20220316_142807" # 1pr (0, 1, 1)
# model_name = "/LSH_model_0.816_20220316_140353" # 1pr (1, 1, 2)
# model_name = "/LSH_model_0.854_20220315_160625"

# Filenames = [ rootpath_save + '/E_TFRecords/dm%s.tfrecords' % a for a in range(6)]
Filenames_3in = [ rootpath_save + '/E_TFRecords/dm%s_3in.tfrecords' % a for a in range(6)]
#Filenames_3in = Filenames_3in[0]
# Weights = [1.0,1.0,1.0,1.0,1.0,1.0]
Weights = [1.0,1.0,1.0,0.0,0.0,0.0]
# Weights = [0.0,0.0,0.0,1.0,1.0,1.0]
jez = hep_model(rootpath_load, rootpath_save, default_filepath, model_name)
# jez.load_tf_model(Filenames_3in, Weights, True, 50, 1000)
jez.prep_for_analysis_tf(Filenames_3in, Weights, True, 1, 100) 

# for element in jez.test_inputs.take(1):
#     print(element)
import pylab as plt
import numpy as np
Pred = jez.prediction
Test = jez.y_test.argmax(axis=1)
def produce_histograms(prediction, flag, no_modes):
    mode_probs_all = [prediction[:,a] for a in range(no_modes)]
    mode_probs = [[],[],[],[],[],[]]
    for a in range(len(prediction)):
        index = flag[a]
        mode_probs[index].append(prediction[a][index])
    fig, ax = plt.subplots(1,no_modes)
    histograms_all = []
    histograms = []
    inv_hists = []
    for a in range(no_modes):
        hist1, _ = np.histogram(mode_probs_all[a], range = (0,1), bins = 30)
        hist2, _ = np.histogram(mode_probs[a], range = (0,1), bins = 30)
        histograms_all.append(hist1)
        histograms.append(hist2)
        inv_hists.append(hist1 - hist2)
    # inv_hist = [a-b for a,b in zip(histograms_all, histograms)]
    print(inv_hists)
    for a in range(no_modes):

        ax[a].hist(np.linspace(0,1,30), weights=inv_hists[a],bins=30)
    # print(mode_probs[a])
    plt.savefig("image.png")
produce_histograms(Pred, Test, 3)
#self.prediction.argmax(axis=1)