print("importing packages...")
import uproot3
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import xgboost as xgb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#~~ Neural Net Stuff ~~#
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import History 
from tensorflow.keras.utils import normalize


#~~ Fin edit ~~#
import vector
import awkward as ak  
import numba as nb
#import ROOT
import time
import sys
#from ROOT import Math, TLorentzVector, TFile
time_start = time.time()
print("loading uproot files...")

treeGG_tt = uproot3.open("/vols/cms/fjo18/Masters2021/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
treeVBF_tt = uproot3.open("/vols/cms/fjo18/Masters2021/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
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
                #"cl_px_1", "cl_py_1", "cl_pz_1", "sc1_Nclusters_1", 
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
                #"cl_px_2", "cl_py_2", "cl_pz_2", "sc1_Nclusters_2", 
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

dfVBF_tt_1 = treeVBF_tt.pandas.df(variables_tt_1)
dfGG_tt_1 = treeGG_tt.pandas.df(variables_tt_1)
dfVBF_tt_2 = treeVBF_tt.pandas.df(variables_tt_2)
dfGG_tt_2 = treeGG_tt.pandas.df(variables_tt_2)
time_elapsed = time_start - time.time()
print("time elapsed = " + str(time_elapsed))

print("Manipulating dataframes...")
time_start = time.time()

df_1 = pd.concat([dfVBF_tt_1,dfGG_tt_1], ignore_index=True) 
df_2 = pd.concat([dfVBF_tt_2,dfGG_tt_2], ignore_index=True) 
#combine gluon and vbf data for hadronic modes
del dfVBF_tt_1, dfVBF_tt_2, dfGG_tt_2

#~~ Separating the tt data into two separate datapoints ~~#

df_1.set_axis(variables_tt_2, axis=1, inplace=True) 
# rename axes to the same as variables 2
df_full = pd.concat([df_1, df_2], ignore_index = True)
del df_1, df_2

df_full = df_full.head(1000000)

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

def fullmomlists(dataframe, fourmomenta_list):
    dataframe["E_full_list"] = dataframe[fourmomenta_list[0]].values.tolist()
    dataframe["px_full_list"] = dataframe[fourmomenta_list[1]].values.tolist()
    dataframe["py_full_list"] = dataframe[fourmomenta_list[2]].values.tolist()
    dataframe["pz_full_list"] = dataframe[fourmomenta_list[3]].values.tolist()


def phi_eta_find(dataframe,fourmomentum_cols):
    fullmomlists(dataframe, fourmom_list)
    fourvect = vector.arr({"px": dataframe[fourmomentum_cols[1]],\
                       "py": dataframe[fourmomentum_cols[2]],\
                       "pz": dataframe[fourmomentum_cols[3]],\
                        "E": dataframe[fourmomentum_cols[0]]})
    tauvisfourvect = vector.obj(px = dataframe[tau_2_4mom[1]],\
                               py = dataframe[tau_2_4mom[2]],\
                               pz = dataframe[tau_2_4mom[3]],\
                               E = dataframe[tau_2_4mom[0]])
    dataframe["phis"] = fourvect.deltaphi(tauvisfourvect) 
    dataframe["etas"] = fourvect.deltaeta(tauvisfourvect) 
    dataframe["frac_energies"] = fourvect.E/tauvisfourvect.E
    #dataframe["full_energies"] = fourvect.E
    # fractional energy
    return #phicoords, etacoords, energy

#phi_eta_find(df_ordered,fourmom_list_colnames )



def largegrid(dataframe,fourmomentum_cols, dimension_l, dimension_s):
    phi_eta_find(dataframe,fourmomentum_cols)
    halfdim = dimension_l/2
    halfdim2 = dimension_s/2
    largegridlist = []
    smallgridlist = []
    imcounter = 0
    counter = 0
    for index, row in dataframe.iterrows():
        if counter <100000:
            grid = np.zeros((dimension_l,dimension_l), float)
            grid2 = np.zeros((dimension_s,dimension_s), float)
            phis = np.array(row["phis"])
            etas = np.array(row["etas"])
            energies = row["frac_energies"]

            # ARRAY SIZES: outer is 21x21, -0.55 to 0.55 in phi/eta
            #              inner is 11x11, -0.1 to 0.1 in phi/eta


            phicoords =  np.floor((phis/1.2) * dimension_l + halfdim).astype(int)
            etacoords =  np.floor(-1 * (etas/1.2) * dimension_l + halfdim).astype(int)
            phicoords2 =  np.floor((phis/0.2) * dimension_s + halfdim2).astype(int)
            etacoords2 =  np.floor(-1 * (etas/0.2) * dimension_s + halfdim2).astype(int)
            for a in range(len(energies)):
                if energies[a] != 0.0:
                    grid[etacoords[a]][phicoords[a]] += energies[a]
                    if etacoords2[a] < dimension_s and etacoords2[a] >= 0 and phicoords2[a] < dimension_s and phicoords2[a] >=0:
                        grid2[etacoords2[a]][phicoords2[a]] += energies[a]
            largegridlist.append(grid)
            smallgridlist.append(grid2)
            counter +=1
        else:
            np.save('/vols/cms/fjo18/Masters2021/Images/image_l_%02d.npy' % imcounter, largegridlist)
            np.save('/vols/cms/fjo18/Masters2021/Images/image_s_%02d.npy' % imcounter, smallgridlist)
            largegridlist = []
            smallgridlist = []
            imcounter+=1
            counter = 0
    np.save('/vols/cms/fjo18/Masters2021/Images/image_l_%02d.npy' % imcounter, largegridlist)
    np.save('/vols/cms/fjo18/Masters2021/Images/image_s_%02d.npy' % imcounter, largegridlist)
    




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

print("Generating image arrays...")
time_start = time.time()

largegrid(df_ordered, fourmom_list_colnames, 21, 11)
time_elapsed = time_start - time.time()
print("elapsed time = " + str(time_elapsed))

#np.save('largeimages.npy', image_l)


#~~FULL LIST OF VARIABLES~~#
# eta: η of τvis 
# pt: pT of τvis
# Mrho: mass(π, S)
# Mrho_OneHighGammas: mass(π, γ/e lead)
# Mrho_subleadingGamma: mass(π, γ/e sublead)
# Mrho_TwoHighGammas: mass(π, γ/elead, γ/e sublead)
# Egamma1_tau: Eγ/e_lead / Evis_τ
# Egamma2_tau:Eγ/e_sublead / Evis_τ
# Mpi0: mass(S)
# Mpi0 TwoHighGammas: mass(γ/e_lead, γ/e_sublead)
# strip pt: pT of S
# Epi0: energy of S
# Epi tau: Eπ / Evis_τ
# Epi: Eπ
# tau decay mode: HPS decay mode
# DeltaR2WRTtau: < ∆R2 > = Σ_i∆R^2_i p^2_Ti / Σ_i p^2_Ti 
# “i” goes over π and all γ/e in S. ∆R is with respect to 
# τvis direction)
# DeltaR2WRTtau tau: < ∆R 2 > ×(E^vis_τ )^2
# rho_dEta: ∆η(π, S)
# rho_dEta_tau: E^vis_τ × ∆η(π, S)
# rho_dphi: ∆φ(π, S)
# rho_dphi_tau: E^vis_τ × ∆φ(π, S)
# gammas_dEta: ∆η(γ/e_lead, γ/e_sublead )
# gammas_dEta_tau: E^vis_τ × ∆η(γ/e_lead, γ/e_sublead )
# gammas_dR_tau: E^vis_τ × ∆R(γ/e_lead, γ/e_sublead )

