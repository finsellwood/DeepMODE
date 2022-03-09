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
sys.path.append("/home/hep/ab7018/CMSSW_10_2_19/DeepMODE/Python_files/A_Pipeline/")

from pipeline_object import pipeline 
import numpy as np
import yaml
import os
import pickle
from array import array
import argparse

def parse_arguments():
    """Tool for easy argument calling in functions"""
    parser = argparse.ArgumentParser(
        description="Apply NN model on ROOT file")
    parser.add_argument(
        "--config-training",
        default="nn_training_config.yaml",	#CREATE THIS
        help="Path to training config file")
    parser.add_argument(
        "--dir-prefix",
        type=str,
        default="ntuple",
        help="Prefix of directories in ROOT file to be annotated.")
    parser.add_argument(
        "input", help="Path to input file, where response will be added.")
    parser.add_argument(
        "tag", help="Tag to be used as prefix of the annotation.")
    parser.add_argument(
        "--tree", default="ntuple", help="Name of trees in the directories.")
    parser.add_argument(
        "--channel", default="tt", help="Name of channel to annotate.")
    parser.add_argument(
        "--model_folder", default="/vols/cms/fjo18/Masters2021/D_Models/Models3_TF/LSH_model_0.759_20220307_155115", help="Folder name where trained model is.") #change this
    parser.add_argument(
        "--era", default="", help="Year to use.")
    parser.add_argument(
        "loadpath", default="/vols/cms/dw515/outputs/SM/MPhysNtuples", help="Path to files to use.")
    parser.add_argument(
        "savepath", default="/vols/cms/ab7018/Masters2021/Annotation", help="Path to save output files.")
    return parser.parse_args()

def parse_config(filename):
    """For loading the config file with yaml"""				
    return yaml.load(open(filename, "r"))
    
#def load_files(filelist):
#    """For loading the data files"""
#    with open(filelist) as f:
#        file_names = f.read().splitlines()
#        # file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
#    return file_names
    
    
def main(args, config):

    #open original file
    file_ = ROOT.TFile("{}".format(args.loadpath)+"/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root", "UPDATE")
    tree = file_.Get(args.tree)
    
    #Book branches for annotation
    response_scores_1 = array("f", [0,0,0]) #want array instead of float - troubles??
    branch_scores_1 = tree.Branch("{}_scores_1".format(
        args.tag), response_scores_1, "{}_scores_1/F".format(args.tag))
    
    response_scores_2 = array("f", [0,0,0])
    branch_scores_2 = tree.Branch("{}_scores_2".format(
        args.tag), response_scores_2, "{}_scores_2/F".format(args.tag))
        
    # Run the event loop
    for i_event in range(tree.GetEntries()):
        tree.GetEntry(i_event)
        
        # Get event number and compute response
        event = int(getattr(tree, "event"))
        
        #create a jesmond
        jesmond = pipeline(args.loadpath, args.savepath) #ideally should take vars from config file
        jesmond2 = pipeline(args.loadpath, args.savepath)
    	    
        #load root files for preprocessing
        jesmond.load_single_event(i_event,1) #change this in the pipeline so its flexible and takes from loadpath
        # jesmond.load_single_event(i) #have to run once for VBF and once for GluGlu
        jesmond2.load_single_event(i_event,2)
    
        #do preprocessing
        jesmond.split_full_by_dm(jesmond.df_full)
        jesmond.split_full_by_dm(jesmond2.df_full)
        jesmond.modify_dataframe(jesmond.df_full)
        jesmond.modify_dataframe(jesmond2.df_full)
        imvar_jesmond = jesmond.create_imvar_dataframe(jesmond.df_full)
        imvar_jesmond2 = jesmond.create_imvar_dataframe(jesmond2.df_full)
        #jesmond.clear_dataframe()          
        test1 = generate_datasets_anal(jesmond.df_full, imvar_jesmond, args.savepath)  #modify this not to save but create 
        test2 = generate_datasets_anal(jesmond2.df_full, imvar_jesmond2, args.savepath)  #modify this not to save but create 
    
        #load our model
        model = keras.models.load_model(args.model_folder)
            
        response_scores_1 = model.predict(test1) #have to properly feed the whole event with the images #.predict_prova?
        response_scores_2 = model.predict(test2)
            
        # Fill branches
        branch_scores_1.Fill()
        branch_scores_2.Fill()
 

    logger.debug("Finished looping over events")

    # Write everything to file
    file_.Write("ntuple",ROOT.TObject.kWriteDelete)
    file_.Close()

    logger.debug("Closed file")           
    
    
                      
if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config_training)
    #file_names = load_files(args.input)
    main(args, config)


