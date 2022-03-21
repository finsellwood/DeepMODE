"""
1. Load a ROOT file
2. Run the preprocessing on it - dataframeinit, dataframemod, imgen
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
print("loaded packages")
def parse_arguments():
    """Tool for easy argument calling in functions"""
    parser = argparse.ArgumentParser(
        description="Apply NN model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="/home/hep/fjo18/CMSSW_10_2_19/src/UserCode/DeepLearning/config.yaml",	#CREATE THIS
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
        "--loadpath", default="/vols/cms/dw515/outputs/SM/Masters2021_ntuples", help="Path to files to use.")
    parser.add_argument(
        "--savepath", default="/vols/cms/fjo18/Masters2021/Annotation", help="Path to save output files.")
    return parser.parse_args()

def parse_config(filename):
    """For loading the config file with yaml"""				
    return yaml.load(open(filename, "r"))

files = ["DY1JetsToLL-LO_tt_2018.root",
"GluGluHToTauTau_M-125_tt_2018.root",
"VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",
"WZTo1L3Nu_tt_2018.root",
"DY2JetsToLL-LO_tt_2018.root",
"GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root",
"W1JetsToLNu-LO_tt_2018.root",                             
"WZTo2L2Q_tt_2018.root",
"DY3JetsToLL-LO_tt_2018.root",
"TauA_tt_2018.root",
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
"TauD_tt_2018.root",
"WGToLNuG_tt_2018.root",
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
"VBFHToTauTau_M-125-ext1_tt_2018.root",
"WZTo1L1Nu2Q_tt_2018.root"]
    
#def load_files(filelist):
#    """For loading the data files"""
#    with open(filelist) as f:
#        file_names = f.read().splitlines()
#        # file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
#    return file_names
    
for file_number in files:   
    def main(args):#, config):

        #open original file
        file_ = ROOT.TFile("{}".format(args.loadpath)+files[file_number], "UPDATE")
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
        
        model_object = hep_model("/vols/cms/fjo18/Masters2021/", "/vols/cms/fjo18/Masters2021/", args.model_folder, args.models[0])
        model_object.create_featuredesc()
        model_object.load_model()

        model_object2 = hep_model("/vols/cms/fjo18/Masters2021/", "/vols/cms/fjo18/Masters2021/", args.model_folder, args.models[1])
        model_object2.create_featuredesc()
        model_object2.load_model()

        # Run the event loop
        for i_event in range(100,200):
            #print(i_event)
            tree.GetEntry(i_event)
            
            # Get event number and compute response
            event = int(getattr(tree, "event"))
            #print(event)
            #create a jesmond
            jesmond = pipeline(args.loadpath, args.savepath) #ideally should take vars from config file
            jesmond2 = pipeline(args.loadpath, args.savepath)
                
            #load root files for preprocessing
            jesmond.load_single_event(i_event,1) #change this in the pipeline so its flexible and takes from loadpath
            # jesmond.load_single_event(i) #have to run once for VBF and once for GluGlu
            jesmond2.load_single_event(i_event,2)
            # for index,row in jesmond.df_full.iterrows():
            #     print(row["pi0_E_2"])
            # print(jesmond.df_full["pi0_E_2"])
            # print(jesmond2.df_full.head(1))
            #print(jesmond.df_full.head())
            #do preprocessing
            # jesmond.modify_dataframe_se(newdf)
            jesmond.modify_dataframe_se(jesmond.df_full)
            jesmond.modify_dataframe_se(jesmond2.df_full)
            # print("got this far") PENISPENIS
            imvar_jesmond = jesmond.create_imvar_dataframe(jesmond.df_full, one_event=True)
            imvar_jesmond2 = jesmond.create_imvar_dataframe(jesmond2.df_full, one_event=True)
            # print(jesmond.df_full)
            #jesmond.clear_dataframe()          
            #test1 = jesmond.generate_datasets_anal_2(jesmond.df_full, imvar_jesmond, args.savepath)  #modify this not to save but create 
            #test2 = jesmond.generate_datasets_anal_2(jesmond2.df_full, imvar_jesmond2, args.savepath)  #modify this not to save but create 
            #print(test1)
            # print(test1.take(1)["hl"])
            #load our model

            # model = keras.models.load_model(args.model_folder)
            
            # jesmond.generate_datasets_anal(jesmond.df_full, imvar_jesmond, args.savepath)  #modify this not to save but create 
            # response_scores_1 = model_object.analyse_event(args.savepath+"/dm.tfrecords")
            raw_ds = jesmond.generate_datasets_anal_4(jesmond.df_full, imvar_jesmond, args.savepath)
            # print(raw_ds)
            if int(getattr(tree, "tau_decay_mode_1"))<10:
                response_1 = model_object.analyse_event_from_raw(raw_ds)
            else: 
                response_1 = model_object2.analyse_event_from_raw(raw_ds)

            raw_ds = jesmond.generate_datasets_anal_4(jesmond2.df_full, imvar_jesmond2, args.savepath)
            if int(getattr(tree, "tau_decay_mode_1"))<10:
                response_2 = model_object.analyse_event_from_raw(raw_ds)
            else: 
                response_2 = model_object2.analyse_event_from_raw(raw_ds)
            response_2 = model_object.analyse_event_from_raw(raw_ds)

            response_0_scores_1[0] = response_1[0][0]
            response_0_scores_2[0] = response_2[0][0] 
            response_1_scores_1[0] = response_1[0][1]
            response_1_scores_2[0] = response_2[0][1] 
            response_2_scores_1[0] = response_1[0][2]
            response_2_scores_2[0] = response_2[0][2] 
            response_10_scores_1[0] = response_1[0][3]
            response_10_scores_2[0] = response_2[0][3] 
            response_11_scores_1[0] = response_1[0][4]
            response_11_scores_2[0] = response_2[0][4] 
            response_other_scores_1[0] = response_1[0][5]
            response_other_scores_2[0] = response_2[0][5] 

            # response_scores_1 = model.predict(args.savepath+"/dm.tfrecords") #have to properly feed the whole event with the images #.predict_prova?
            # jesmond.generate_datasets_anal_4(jesmond2.df_full, imvar_jesmond2, args.savepath)
            # response_scores_2 = model.predict(args.savepath+"/dm.tfrecords")
            # response_scores_2 = model_object.analyse_event(args.savepath+"/dm.tfrecords")

            # max_1 = max([response_0_scores_1,response_1_scores_1,response_2_scores_1,response_10_scores_1,response_11_scores_1])
            # max_2 = max([response_0_scores_2,response_1_scores_2,response_2_scores_2,response_10_scores_2,response_11_scores_2])
            # if max_1 == response_0_scores_1:
            #     indicator1 = 0
            # elif max_1 == response_1_scores_1:
            #     indicator1 = 1
            # elif max_1 == response_2_scores_1:
            #     indicator1 = 2
            # elif max_1 == response_10_scores_1:
            #     indicator1 == 10
            # elif max_1 == response_11_scores_1:
            #     indicator1 == 11
            # else: indicator1 = -1
            # print(indicator1 == tree.tauFlag_1)
            
            # if max_2 == response_0_scores_2:
            #     indicator2 = 0
            # elif max_2 == response_1_scores_2:
            #     indicator2 = 1
            # elif max_2 == response_2_scores_2:
            #     indicator2 = 2
            # elif max_2 == response_10_scores_2:
            #     indicator2 == 10
            # elif max_2 == response_11_scores_2:
            #     indicator2 == 11
            # else: indicator2 == -1
            # print(indicator2 == tree.tauFlag_2)
            #response_scores_1 = model.predict(test1)
            #response_scores_2 = model.predict(test2)
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

        logger.debug("Finished looping over events")

        # Write everything to file
        file_.Write("ntuple",ROOT.TObject.kWriteDelete)
        file_.Close()

        logger.debug("Closed file")           
    
    
                      
if __name__ == "__main__":
    args = parse_arguments()
    #config = parse_config(args.config_training)
    #file_names = load_files(args.input)
    main(args)#, config)


