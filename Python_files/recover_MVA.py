#~~ DATAFRAMEINIT.PY ~~#
# Unpacks root files into a dataframe, selecting necessary columns in process.
# Generates new variables and appends them to the dataframe as new columns before saving df_ordered as ordereddf.pkl
# Also recovers y values and saves in a separate dataframe as yvalues.pkl
rootpath = "/vols/cms/fjo18/Masters2021"
print("importing packages...")
import uproot3
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import sys

time_start = time.time()
print("loading uproot files...")

treeGG_tt = uproot3.open(rootpath + "/RootFiles/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
treeVBF_tt = uproot3.open(rootpath + "/RootFiles/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]

y = pd.read_pickle(rootpath + "/Objects/yvalues.pkl")
# Y array so that the data can be split properly/double-checked
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
                "mva_dm_1",
               ]

variables_tt_2 = ["tauFlag_2", 
                "mva_dm_2", 
               ]


time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
print("converting root files to dataframes...")
time_start = time.time()

dfVBF_tt_1 = treeVBF_tt.pandas.df(variables_tt_1)
dfGG_tt_1 = treeGG_tt.pandas.df(variables_tt_1)
dfVBF_tt_2 = treeVBF_tt.pandas.df(variables_tt_2)
dfGG_tt_2 = treeGG_tt.pandas.df(variables_tt_2)


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



df_mva = pd.concat([df_DM0, df_DM1, df_DM2, df_DM10, df_DM11, df_DMminus1], ignore_index = True)
del df_DM0, df_DM1, df_DM2, df_DMminus1, df_DM10, df_DM11, df_full



time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))

print(y.sample(n = 5,  random_state= 1234))
print(df_mva.sample(n = 5,  random_state= 1234))
# double check they're the same

mva_train, mva_test = train_test_split(
    df_mva,
    test_size=0.2,
    random_state=123456,
    stratify = y
)

mva_train.to_pickle(rootpath + "/Objects/mva_train.pkl")
mva_test.to_pickle(rootpath + "/Objects/mva_test.pkl")

