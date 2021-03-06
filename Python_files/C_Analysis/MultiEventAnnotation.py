"""
1. Load a ROOT file
2. Run the preprocessing on all the data - dataframeinit, dataframemod, imgen
3. Run analyse model pointing to this data
4. Annotate branches

Have a config file that sets configs
"""

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import logging
logger = logging.getLogger("annotate_file_inc.py")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler) 
import sys
#sys.path.append("/home/hep/ab7018/CMSSW_10_2_19/DeepMODE//Python_files/A_Pipeline/")
#sys.path.append("/home/hep/ab7018/CMSSW_10_2_19/DeepMODE/Python_files/B_Training/")
sys.path.append("/home/hep/fjo18/CMSSW_10_2_19/src/UserCode/DeepLearning/Python_files/A_Pipeline/")
sys.path.append("/home/hep/fjo18/CMSSW_10_2_19/src/UserCode/DeepLearning/Python_files/B_Training/")

from pipeline_object import pipeline 
from model_object import hep_model
import numpy as np
import yaml
import os
import pickle
from array import array
import argparse
from tensorflow import keras
import time
print("loaded packages")
def parse_arguments():
    """Tool for easy argument calling in functions"""
    parser = argparse.ArgumentParser(
        description="Apply NN model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="/home/hep/ab7018/CMSSW_10_2_19/DeepMODE/config.yaml",	#CREATE THIS
        help="Path to training config file")
    parser.add_argument(
        "--dir-prefix",
        type=str,
        default="ntuple",
        help="Prefix of directories in ROOT file to be annotated.")
#    parser.add_argument(
#        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "--tag", default="1", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--channel", default="tt", help="Name of channel to annotate.")
    parser.add_argument(
        "--model_folder", default="/D_Models/Models3_TF/", help="Folder name where trained model is.") #change this
    parser.add_argument(
        "--models", type=str, nargs="+",
         default=["LSH_model_0.759_20220307_155115", "LSH_model_0.852_20220315_160854"], 
         help="Names of trained models") #last changed 21.03.2022
    parser.add_argument(
        "--era", default="", help="Year to use.")
    parser.add_argument(
        "--loadpath", default="/vols/cms/fjo18/Masters2021/RootFiles/Full", help="Path to files to use.")
    parser.add_argument(
        "--savepath", default="/vols/cms/fjo18/Masters2021/Annotation", help="Path to save output files.")
    return parser.parse_args()

def parse_config(filename):
    """For loading the config file with yaml"""				
    return yaml.load(open(filename, "r"))

files = ["DY1JetsToLL-LO_tt_2018.root",
"GluGluHToTauTau_M-125_tt_2018.root",
"VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",# checked 15.04.2022 09.02
"WZTo1L3Nu_tt_2018.root",
"DY2JetsToLL-LO_tt_2018.root", # 12.04.2022 4.40 checked
"GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",
"W1JetsToLNu-LO_tt_2018.root",                             
"WZTo2L2Q_tt_2018.root",
"DY3JetsToLL-LO_tt_2018.root",
"TauA_tt_2018.root", # checked 15.04.2022 10.42 
"W2JetsToLNu-LO_tt_2018.root",
"WZTo3LNu-ext1_tt_2018.root",
"DY4JetsToLL-LO_tt_2018.root",
"TauB_tt_2018.root",
"W3JetsToLNu-LO_tt_2018.root",
"WZTo3LNu_tt_2018.root",
"DYJetsToLL-LO_tt_2018.root",
"TauC_tt_2018.root",
"W4JetsToLNu-LO_tt_2018.root",
"ZHToTauTau_M-125_tt_2018.root",
"DYJetsToLL_M-10-50-LO_tt_2018.root",
"TauD_tt_2018.root", # 15.04.2022 22.06 checked 
"WGToLNuG_tt_2018.root", # 14.04.2022 4.57 checked
"ZHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",
"DYJetsToLL_tt_2018.root",
"Tbar-t_tt_2018.root",
"WJetsToLNu-LO_tt_2018.root",
"ZZTo2L2Nu-ext1_tt_2018.root",
"EmbeddingTauTauA_tt_2018.root",
"Tbar-tW-ext1_tt_2018.root",
"WminusHToTauTau_M-125_tt_2018.root",
"ZZTo2L2Nu-ext2_tt_2018.root",
"EmbeddingTauTauB_tt_2018.root",
"TTTo2L2Nu_tt_2018.root",
"WminusHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",
"ZZTo2L2Q_tt_2018.root",
"EmbeddingTauTauC_tt_2018.root",
"TTToHadronic_tt_2018.root",
"WplusHToTauTau_M-125_tt_2018.root",
"ZZTo4L-ext_tt_2018.root",
"EmbeddingTauTauD_tt_2018.root",
"TTToSemiLeptonic_tt_2018.root",
"WplusHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",
"ZZTo4L_tt_2018.root",
"EWKWMinus2Jets_tt_2018.root",
"T-t_tt_2018.root",
"WWTo2L2Nu_tt_2018.root",
"EWKWPlus2Jets_tt_2018.root",
"T-tW-ext1_tt_2018.root",
"WWToLNuQQ_tt_2018.root",
"EWKZ2Jets_tt_2018.root",
"VBFHToTauTau_M-125-ext1_tt_2018.root", # 14.04.2022 19.12 checked
"WZTo1L1Nu2Q_tt_2018.root"
]

    
  
def main(args):#, config):
    for file_name in files: 
        logger.debug("Updating " + file_name)
        time_start = time.time()
        
        model_object = hep_model("/vols/cms/fjo18/Masters2021/", "/vols/cms/fjo18/Masters2021/", args.model_folder, args.models[0])
        model_object.create_featuredesc()
        model_object.load_model()

        model_object2 = hep_model("/vols/cms/fjo18/Masters2021/", "/vols/cms/fjo18/Masters2021/", args.model_folder, args.models[1])
        model_object2.create_featuredesc()
        model_object2.load_model()

        #create a jesmond
        jesmond = pipeline(args.loadpath, args.savepath)
        jesmond2 = pipeline(args.loadpath, args.savepath)
        #load root files for preprocessing
        jesmond.save_dataframes(1,file_name)
        jesmond2.save_dataframes(2,file_name)

        time_saved = time.time() - time_start
        print("Processed in"+ str(time_saved))

        response_1 = []
        for i in range(6): #jesmond.no_blocks
            jesmond.load_dataframes(1,int((i+1)*1e6))
            jesmond.modify_dataframe(jesmond.df_full)
            imvar_jesmond = jesmond.create_imvar_dataframe(jesmond.df_full, one_event=False)
            raw_ds = jesmond.generate_dataframes_anal_multi(jesmond.df_full, imvar_jesmond, args.savepath)

            time_proc = time.time() - time_start
            print("Processed block %s in" % i + str(time_proc))
            mask = jesmond.decay_mode
            #calculate scores for all events with both classifiers
            response_1_1pr = model_object.predict_results_one_tf(raw_ds)
            response_1_3pr = model_object2.predict_results_one_tf(raw_ds)
            # now filter the scores using the mask
            for j in range(len(mask)):
                if mask[j] < 10: 
                    response_1.append(response_1_1pr[j])
                else: response_1.append(response_1_3pr[j])
            time_eval = time.time() - time_start
            print("Evaluated block %s in " % i + str(time_eval))

        with open("/vols/cms/fjo18/Masters2021/Annotation/response_1", "wb") as f:
            pickle.dump(response_1, f)

        del jesmond, imvar_jesmond, response_1_1pr, response_1_3pr, raw_ds, mask

        response_2 = []
        for i in range(6): 
            jesmond2.load_dataframes(2,int((i+1)*1e6))
            jesmond2.modify_dataframe(jesmond2.df_full)
            imvar_jesmond2 = jesmond2.create_imvar_dataframe(jesmond2.df_full, one_event=False)
            raw_ds_2 = jesmond2.generate_dataframes_anal_multi(jesmond2.df_full, imvar_jesmond2, args.savepath)

            time_proc = time.time() - time_start
            print("Processed second tau block %s in" % i + str(time_proc))
            mask2 = jesmond2.decay_mode
            #calculate scores for all events with both classifiers
            response_2_1pr = model_object.predict_results_one_tf(raw_ds_2)
            response_2_3pr = model_object2.predict_results_one_tf(raw_ds_2)
            # now filter the scores using the mask
            for j in range(len(mask2)):
                if mask2[j] < 10: 
                    response_2.append(response_2_1pr[j])
                else: response_2.append(response_2_3pr[j])
            time_eval = time.time() - time_start
            print("Evaluated second tau block %s in " % i + str(time_eval))

        with open("/vols/cms/fjo18/Masters2021/Annotation/response_2", "wb") as f:
            pickle.dump(response_2, f)

        del jesmond2, imvar_jesmond2, response_2_1pr, response_2_3pr, raw_ds_2, mask2

        #open original file
        file_ = ROOT.TFile("{}".format(args.loadpath)+"/"+file_name, "UPDATE")
        tree = file_.Get(args.tree)

        #Book branches for annotation
        response_0_scores_1 = array("f", [-9999]) 
        branch_0_scores_1 = tree.Branch("{}_score_1".format(
                0), response_0_scores_1, "{}_score_1/F".format(0))
    
        response_0_scores_2 = array("f", [-9999])
        branch_0_scores_2 = tree.Branch("{}_score_2".format(
            0), response_0_scores_2, "{}_score_2/F".format(0))
    
        response_1_scores_1 = array("f", [-9999]) 
        branch_1_scores_1 = tree.Branch("{}_score_1".format(
                1), response_1_scores_1, "{}_score_1/F".format(1))
    
        response_1_scores_2 = array("f", [-9999])
        branch_1_scores_2 = tree.Branch("{}_score_2".format(
            1), response_1_scores_2, "{}_score_2/F".format(1))
    
        response_2_scores_1 = array("f", [-9999]) 
        branch_2_scores_1 = tree.Branch("{}_score_1".format(
                2), response_2_scores_1, "{}_score_1/F".format(2))
        
        response_2_scores_2 = array("f", [-9999])
        branch_2_scores_2 = tree.Branch("{}_score_2".format(
            2), response_2_scores_2, "{}_score_2/F".format(2))

        response_10_scores_1 = array("f", [-9999]) 
        branch_10_scores_1 = tree.Branch("{}_score_1".format(
                10), response_10_scores_1, "{}_score_1/F".format(10))
        
        response_10_scores_2 = array("f", [-9999])
        branch_10_scores_2 = tree.Branch("{}_score_2".format(
            10), response_10_scores_2, "{}_score_2/F".format(10))

        response_11_scores_1 = array("f", [-9999]) 
        branch_11_scores_1 = tree.Branch("{}_score_1".format(
                11), response_11_scores_1, "{}_score_1/F".format(11))
        
        response_11_scores_2 = array("f", [-9999])
        branch_11_scores_2 = tree.Branch("{}_score_2".format(
            11), response_11_scores_2, "{}_score_2/F".format(11))

        response_other_scores_1 = array("f", [-9999]) 
        branch_other_scores_1 = tree.Branch("{}_score_1".format(
                "other"), response_other_scores_1, "{}_score_1/F".format("other"))
        
        response_other_scores_2 = array("f", [-9999])
        branch_other_scores_2 = tree.Branch("{}_score_2".format(
            "other"), response_other_scores_2, "{}_score_2/F".format("other"))

        for i_event in range(tree.GetEntries()):
            #print(i_event)
            tree.GetEntry(i_event)
            if i_event % 10000 == 0:
                    logger.debug('Currently on event {}'.format(i_event))

            #does the indexing hold? i dont think so
            response_0_scores_1[0] = response_1[i_event][0]
            response_0_scores_2[0] = response_2[i_event][0] 
            response_1_scores_1[0] = response_1[i_event][1]
            response_1_scores_2[0] = response_2[i_event][1] 
            response_2_scores_1[0] = response_1[i_event][2]
            response_2_scores_2[0] = response_2[i_event][2] 
            response_10_scores_1[0] = response_1[i_event][3]
            response_10_scores_2[0] = response_2[i_event][3] 
            response_11_scores_1[0] = response_1[i_event][4]
            response_11_scores_2[0] = response_2[i_event][4] 
            response_other_scores_1[0] = response_1[i_event][5]
            response_other_scores_2[0] = response_2[i_event][5] 

            # Fill branches
            branch_0_scores_1.Fill()
            branch_0_scores_2.Fill()
            branch_1_scores_1.Fill()
            branch_1_scores_2.Fill()
            branch_2_scores_1.Fill()
            branch_2_scores_2.Fill()
            branch_10_scores_1.Fill()
            branch_10_scores_2.Fill()
            branch_11_scores_1.Fill()
            branch_11_scores_2.Fill()
            branch_other_scores_1.Fill()
            branch_other_scores_2.Fill()

        time_filled = time.time() - time_start
        print("Filled in"+ str(time_filled))
        logger.debug("Finished looping over events")

        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()

        logger.debug("Closed "+file_name)           
    
    
                      
if __name__ == "__main__":
    args = parse_arguments()
    #config = parse_config(args.config_training)
    #file_names = load_files(args.input)
    main(args)#, config)


