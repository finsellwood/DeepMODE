#~~ DATAFRAMEINIT.PY ~~#
# Unpacks root files into a dataframe, selecting necessary columns in process.
# Generates new variables and appends them to the dataframe as new columns before saving df_ordered as ordereddf.pkl
# Also recovers y values and saves in a separate dataframe as yvalues.pkl

print("importing packages...")
#import uproot3
import ROOT
import root_numpy as rnp
import numpy as np
import pandas as pd
import vector
import awkward as ak  
import numba as nb
import time
import sys

time_start = time.time()
print("loading uproot files...")

rootGG_tt = ROOT.TFile("/vols/cms/fjo18/Masters2021/RootFiles/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
intreeGG_tt = rootGG_tt.Get("ntuple")
rootVBF_tt = ROOT.TFile("/vols/cms/fjo18/Masters2021/RootFiles/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")
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
               ]

# List labels for later:
pi_1_4mom = ["pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", ]
pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
pi2_1_4mom = ["pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1", ]
pi2_2_4mom = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2", ]
pi3_1_4mom = ["pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1", ]
pi3_2_4mom = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2", ]

pi0_1_4mom = ["pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1", ]
pi0_2_4mom = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", ]

gam1_1_4mom = ["gam1_E_1", "gam1_px_1", "gam1_py_1", "gam1_pz_1", ]
gam2_1_4mom = ["gam2_E_1", "gam2_px_1", "gam2_py_1", "gam2_pz_1", ]
gam1_2_4mom = ["gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2", ]
gam2_2_4mom = ["gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2", ]

sc1_2_4mom = ["sc1_E_2", "sc1_px_2", "sc1_py_2", "sc1_pz_2", ]
# 4-mom of the supercluster

gam_2_3mom = ["gam_E_2", "gam_px_2", "gam_py_2", "gam_pz_2", ]
# 3-momentum of photons as vectors NOTE: gam_E_2 is not defined in the original dataframe but created after the fact

cl_2_3mom = ["sc1_Nclusters_2", "cl_px_2", "cl_py_2", "cl_pz_2", ]
# 3-momentum of supercluster components

tau_1_4mom = ["tau_E_1", "tau_px_1", "tau_py_1", "tau_pz_1", ]
tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]
# Visible tau components


def energyfinder(dataframe, momvariablenames_1):
    momvect1 = vector.arr({"px": dataframe[momvariablenames_1[1]],\
                       "py": dataframe[momvariablenames_1[2]],\
                       "pz": dataframe[momvariablenames_1[3]]})
    dataframe[momvariablenames_1[0]] = abs(momvect1)
# This function generates the energy of the gammas (magnitude of 3-mom) 
# So that they can be treated the same as the other four-momenta later
time_elapsed = time_start - time.time()
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

time_elapsed = time_start - time.time()
print("elapsed time = " + str(time_elapsed))

print("producing new variables...")
time_start = time.time()

#~~ define sets of variables necessary ~~#
measured4mom = [pi_2_4mom, pi2_2_4mom, pi3_2_4mom, gam_2_3mom, sc1_2_4mom, ]

E_list = [a[0] for a in measured4mom]
px_list = [a[1] for a in measured4mom]
py_list = [a[2] for a in measured4mom]
pz_list = [a[3] for a in measured4mom]

# lists of the column names for each component of a four vector
fourmom_list = [E_list, px_list, py_list, pz_list]
# a list of strings pointing to columns with all fourmomenta in
fourmom_list_colnames = ["E_full_list", "px_full_list", "py_full_list", "pz_full_list"]
# a list of actual columns with the lists of fourmomenta in

#~~ Amend data

energyfinder(df_ordered, gam_2_3mom)
#firstlayer = df_ordered.head(1)


def phi_eta(dataframe,momvariablenames_1):
    fourvect = vector.arr({"px": dataframe[momvariablenames_1[1]],\
                       "py": dataframe[momvariablenames_1[2]],\
                       "pz": dataframe[momvariablenames_1[3]],\
                        "E": dataframe[momvariablenames_1[0]]})
    tauvisfourvect = vector.arr({"px": dataframe[tau_2_4mom[1]],\
                               "py": dataframe[tau_2_4mom[2]],\
                               "pz": dataframe[tau_2_4mom[3]],\
                               "E": dataframe[tau_2_4mom[0]]})
    phi = fourvect.deltaphi(tauvisfourvect) 
    eta = fourvect.deltaeta(tauvisfourvect) 
    energy = fourvect.E/tauvisfourvect.E
    # fractional energy
    return phi, eta, energy

def inv_mass(Energ,Px,Py,Pz):
    vect = vector.obj(px=Px, py=Py, pz=Pz, E=Energ)
    return vect.mass

df_ordered["pi0_2mass"] = inv_mass(df_ordered["pi0_E_2"],df_ordered["pi0_px_2"],df_ordered["pi0_py_2"],df_ordered["pi0_pz_2"]) #pion masses

def rho_mass(dataframe, momvariablenames_1, momvariablenames_2):
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                       py = dataframe[momvariablenames_2[2]],\
                       pz = dataframe[momvariablenames_2[3]],\
                       E = dataframe[momvariablenames_2[0]])
    rho_vect = momvect1+momvect2
    dataframe["rho_mass"] = inv_mass(rho_vect.E, rho_vect.px, rho_vect.py, rho_vect.pz) #rho masses
    
rho_mass(df_ordered, pi_2_4mom, pi0_2_4mom)
# rho mass is the addition of the four-momenta of the charged and neutral pions

df_ordered["E_gam/E_tau"] = df_ordered["gam1_E_2"].divide(df_ordered["tau_E_2"]) #Egamma/Etau
df_ordered["E_pi/E_tau"] = df_ordered["pi_E_2"].divide(df_ordered["tau_E_2"]) #Epi/Etau
df_ordered["E_pi0/E_tau"] = df_ordered["pi0_E_2"].divide(df_ordered["tau_E_2"]) #Epi0/Etau

def tau_eta(dataframe, momvariablenames_1):
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    dataframe["tau_eta"] = momvect1.eta  #tau eta (tau pt just a variable)
    
tau_eta(df_ordered, tau_2_4mom)

def ang_var(dataframe, momvariablenames_1, momvariablenames_2, particlename): #same for gammas and pions
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                       py = dataframe[momvariablenames_2[2]],\
                       pz = dataframe[momvariablenames_2[3]],\
                       E = dataframe[momvariablenames_2[0]])
    
    diffphi = momvect1.phi - momvect2.phi
    diffeta = momvect1.eta - momvect2.eta
    diffr = np.sqrt(diffphi**2 + diffeta**2)
    Esum = dataframe[momvariablenames_1[0]] + dataframe[momvariablenames_2[0]]
    dataframe["delR_"+ particlename] = diffr
    dataframe["delPhi_"+ particlename] = diffphi
    dataframe["delEta_" + particlename] = diffeta
    dataframe["delR_xE_"+ particlename] = diffr * Esum
    dataframe["delPhi_xE_"+ particlename] = diffphi * Esum
    dataframe["delEta_xE_" + particlename] = diffeta * Esum
        
ang_var(df_ordered, gam1_2_4mom, gam2_2_4mom, "gam")
ang_var(df_ordered, pi0_2_4mom, pi_2_4mom, "pi")

time_elapsed = time_start - time.time()
print("elapsed time = " + str(time_elapsed))

df_ordered.to_pickle("/vols/cms/fjo18/Masters2021/Objects/ordereddf.pkl")
y.to_pickle("/vols/cms/fjo18/Masters2021/Objects/yvalues.pkl")
