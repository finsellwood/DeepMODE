#~~ DATAFRAMEINIT.PY ~~#
# Unpacks root files into a dataframe, selecting necessary columns in process.
# Generates new variables and appends them to the dataframe as new columns before saving df_ordered as ordereddf.pkl
# Also recovers y values and saves in a separate dataframe as yvalues.pkl
rootpath = "/vols/cms/fjo18/Masters2021"
print("importing packages...")
import ROOT
import root_numpy as rnp
import numpy as np
import pandas as pd
import time
import sys

time_start = time.time()
print("loading uproot files...")

rootGG_tt = ROOT.TFile(rootpath + "/RootFiles/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
intreeGG_tt = rootGG_tt.Get("ntuple")
rootVBF_tt = ROOT.TFile(rootpath + "/RootFiles/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
intreeVBF_tt = rootVBF_tt.Get("ntuple")
time_elapsed = time_start - time.time()
print("elapsed time = " + str(time_elapsed))

print("Loading lists...")
time_start = time.time()

#~~ Variables to use ~~#
# Need mass of rho and pi0
# E_gamma/E_tauvis for leading photon

# TO ACTUALLY INCLUDE
# Generator Level:
    # tauflag_1, tauflag_2 (for leading subleading)
    # 
# Visible tau 4-momentum

variables_tt_1 = ["tauFlag_1",
                # Generator-level properties, actual decay mode of taus for training
                "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1",
                "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1",
                "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1",
                # 4-momenta of the charged pions
                "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1", 
                # 4-momenta of neutral pions
                "gam1_px_1", "gam1_py_1", "gam1_pz_1", "gam1_E_1",
                "gam2_px_1", "gam2_py_1", "gam2_pz_1", "gam2_E_1",
                # 4-momenta of two leading photons
                "gam_px_1", "gam_py_1", "gam_pz_1", "n_gammas_1",
                # 3-momenta vectors of all photons
                "sc1_px_1", "sc1_py_1", "sc1_pz_1", "sc1_E_1",
                # 4-momentum of the supercluster
                "cl_px_1", "cl_py_1", "cl_pz_1", "sc1_Nclusters_1",
                # 3-momenta of clusters in supercluster
                "tau_px_1", "tau_py_1", "tau_pz_1", "tau_E_1",
                # 4-momenta of 'visible' tau
                "tau_decay_mode_1",
                #HPS algorithm decay mode
                "pt_1",
                "n_gammas_1",
               ]

variables_tt_2 = ["tauFlag_2", 
                # Generator-level properties, actual decay mode of taus for training
                "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2", 
                "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
                "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
                # 4-momenta of the charged pions
                # Note: pi2/pi3 only apply for 3pr modes
                "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2", 
                # 4-momenta of neutral pions
                "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                # 4-momenta of two leading photons
                "gam_px_2", "gam_py_2", "gam_pz_2", "n_gammas_2",
                # 3-momenta vectors of all photons
                "sc1_px_2", "sc1_py_2", "sc1_pz_2", "sc1_E_2",
                # 4-momentum of the supercluster
                "cl_px_2", "cl_py_2", "cl_pz_2", "sc1_Nclusters_2",
                # 3-momenta of clusters in supercluster
                "tau_px_2", "tau_py_2", "tau_pz_2", "tau_E_2",
                # 4-momenta of 'visible' tau
                "tau_decay_mode_2", 
                # HPS algorithm decay mode
                "pt_2",
                "n_gammas_2",
               ]


time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
print("converting root files to dataframes...")
time_start = time.time()

# Extracting using root_numpy instead
arrVBF_tt_1 = rnp.tree2array(intreeVBF_tt,branches=variables_tt_1)
arrGG_tt_1 = rnp.tree2array(intreeGG_tt,branches=variables_tt_1)
arrVBF_tt_2 = rnp.tree2array(intreeVBF_tt,branches=variables_tt_2)
arrGG_tt_2 = rnp.tree2array(intreeGG_tt,branches=variables_tt_2)

dfVBF_tt_1 = pd.DataFrame(arrVBF_tt_1)
dfGG_tt_1 = pd.DataFrame(arrGG_tt_1)
dfVBF_tt_2 = pd.DataFrame(arrVBF_tt_2)
dfGG_tt_2 = pd.DataFrame(arrGG_tt_2)

time_elapsed = time.time() - time_start
print("time elapsed = " + str(time_elapsed))
print("Manipulating dataframes...")
time_start = time.time()

df_1 = pd.concat([dfVBF_tt_1,dfGG_tt_1], ignore_index=True) 
df_2 = pd.concat([dfVBF_tt_2,dfGG_tt_2], ignore_index=True) 
#combine gluon and vbf data for hadronic modes
del dfVBF_tt_1, dfVBF_tt_2, dfGG_tt_2, dfGG_tt_1

#~~ Separating the tt data into two separate datapoints ~~#

df_1.set_axis(variables_tt_2, axis=1, inplace=True) 
# rename axes to the same as variables 2
df_full = pd.concat([df_1, df_2], ignore_index = True)
del df_1, df_2

#df_full = df_full.head(1000000)

#~~ Filter decay modes and add in order ~~#
df_DM0 = df_full[
      (df_full["tauFlag_2"] == 0)
]
lenDM0 = df_DM0.shape[0]

df_DM1 = df_full[
      (df_full["tauFlag_2"] == 1)
]
lenDM1 = df_DM1.shape[0]

df_DM2 = df_full[
      (df_full["tauFlag_2"] == 2)
]
lenDM2 = df_DM2.shape[0]

df_DM10 = df_full[
      (df_full["tauFlag_2"] == 10)
]
lenDM10 = df_DM10.shape[0]

df_DM11 = df_full[
      (df_full["tauFlag_2"] == 11)
]
lenDM11 = df_DM11.shape[0]

df_DMminus1 = df_full[
      (df_full["tauFlag_2"] == -1)
]
lenDMminus1 = df_DMminus1.shape[0]



df_ordered = pd.concat([df_DM0, df_DM1, df_DM2, df_DM10, df_DM11, df_DMminus1], ignore_index = True)
del df_DM0, df_DM1, df_DM2, df_DMminus1, df_DM10, df_DM11, df_full

d_DM0 = pd.DataFrame({'col': np.zeros(lenDM0)})
d_DM1 = pd.DataFrame({'col': np.ones(lenDM1)})
d_DM2 = pd.DataFrame({'col': 2 * np.ones(lenDM2)})
d_DM10 = pd.DataFrame({'col': 3 * np.ones(lenDM10)})
d_DM11 = pd.DataFrame({'col': 4 * np.ones(lenDM11)})
d_DMminus1 = pd.DataFrame({'col': 5 * np.ones(lenDMminus1)})
y = pd.concat([d_DM0, d_DM1, d_DM2, d_DM10, d_DM11, d_DMminus1], ignore_index = True)

del d_DM0, d_DM1, d_DM2, d_DM10, d_DM11, d_DMminus1

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))


df_ordered.to_pickle(rootpath + "/Objects/ordereddf.pkl")
y.to_pickle(rootpath + "/Objects/yvalues.pkl")
