#~~ DATAFRAMEMOD.PY ~~#
# Updated and functioning (as of 7.12.21)
# Takes the existing n-tuple dataframe (df_ordered) and uses data to create an additional dataframe (imvar_df)
# with only the variables necessary for image creation. Then removes the columns used from df_ordered and saves
# (under diff name)
# This is necessary as with the addition of columns of data (a previously bugged feature) the dataframes take a 
# ridiculous time to load. Removing the lists (which aren't used for training anyway) is important
# Now (22/1/22) this file also rotates the generated coordinates s.t. the charged and neutral pion are on/parallel to the axis

rootpath_load = "/vols/cms/fjo18/Masters2021/Objects3"
rootpath_save = "/vols/cms/fjo18/Masters2021/Objects3"
# Objects3 is for new data, which contains the SC HL vars and comes from dw515 dataframes

load_full = True

#~~ Store strings in memory ~~#

gam1_2_4mom = ["gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2", ]
gam2_2_4mom = ["gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2", ]

pi0_2_4mom = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", ]
pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
pi2_2_4mom = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2", ]
pi3_2_4mom = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2", ]

gam_2_3mom = ["gam_E_2", "gam_px_2", "gam_py_2", "gam_pz_2", ]
# 3-momentum of photons as vectors NOTE: gam_E_2 is not defined in the original dataframe but created after the fact
cl_2_3mom = ["cl_E_2", "cl_px_2", "cl_py_2", "cl_pz_2", ]
# 3-mom of the clusters as lists...hopefully. As with gam_2_3mom, the energy label here is not defined until later

sc1_2_4mom = ["sc1_E_2", "sc1_px_2", "sc1_py_2", "sc1_pz_2", ]
tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]

measured4mom = [pi0_2_4mom, pi_2_4mom, pi2_2_4mom, pi3_2_4mom, sc1_2_4mom, gam_2_3mom, cl_2_3mom, ]
# Updated to include pi0 (previously excluded for some reason (13.12.21))
E_list = [a[0] for a in measured4mom]
px_list = [a[1] for a in measured4mom]
py_list = [a[2] for a in measured4mom]
pz_list = [a[3] for a in measured4mom]
fourmom_list = [E_list, px_list, py_list, pz_list]
# a list of actual columns with the lists of fourmomenta in
fourmom_list_colnames = ["E_full_list", "px_full_list", "py_full_list", "pz_full_list"]


import pandas as pd
import numpy as np
# from pyrsistent import v
import vector
import awkward as ak  
import numba as nb
import time
import matplotlib.pyplot as plt

#~~ Load in pickled dataframe ~~#
print("loading dataframes...")
time_start = time.time()

if load_full:
    df_ordered = pd.read_pickle(rootpath_load + "/ordereddf.pkl")
    print("loading with full array")
else:
    df_ordered = pd.read_pickle(rootpath_load + "/testhead.pkl")
    print("loading with small test array")

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
print("Finding HL variables...")
time_start = time.time()

def energyfinder(dataframe, momvariablenames_1):
    arr = []
    counter = -1
    for i in dataframe[momvariablenames_1[1]]:
        counter += 1
        energ = []
        if i.size > 0:
            for j in range(len(i)):
                momvect_j = [dataframe[momvariablenames_1[1]][counter][j],\
                             dataframe[momvariablenames_1[2]][counter][j],\
                             dataframe[momvariablenames_1[3]][counter][j]]
                energ.append(np.sqrt(sum(x**2 for x in momvect_j)))
            arr.append(energ)
        else: arr.append([])
    dataframe[momvariablenames_1[0]] = arr
# This function generates the energy of the given massless object (magnitude of 3-mom) 
# So that can be treated the same as the other four-momenta later
# oh wow this is bad
def energyfinder_2(dataframe, momvariablenames_1):
    

energyfinder(df_ordered, gam_2_3mom)
energyfinder(df_ordered, cl_2_3mom)
# define the energy variables for the photon and cluster lists


def inv_mass(Energ,Px,Py,Pz):
    vect = vector.obj(px=Px, py=Py, pz=Pz, E=Energ)
    return vect.mass

df_ordered["pi0_2mass"] = inv_mass(df_ordered["pi0_E_2"],df_ordered["pi0_px_2"],df_ordered["pi0_py_2"],df_ordered["pi0_pz_2"]) #pion masses

def rho_mass(dataframe, momvariablenames_1, momvariablenames_2 = None):
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    if momvariablenames_2 is not None:
        momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                        py = dataframe[momvariablenames_2[2]],\
                        pz = dataframe[momvariablenames_2[3]],\
                        E = dataframe[momvariablenames_2[0]])
        rho_vect = momvect1+momvect2
    else:
        rho_vect = momvect1
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

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
print("Finding phis and etas")
time_start = time.time()


def get_fourmomenta_lists(dataframe):
    for a in range(4):
        dataframe[fourmom_list_colnames[a]] = dataframe[fourmom_list[a][0]].apply(lambda x: np.array([x]).flatten().tolist())
        for b in range(1, len(fourmom_list[1])):   
            dataframe[fourmom_list_colnames[a]] += dataframe[fourmom_list[a][b]].apply(lambda x: np.array([x]).flatten().tolist())
      # This method means that the final lists are flat (i.e. clusters/photons dont have a nested structure)
        
def phi_eta_find(dataframe):  
    get_fourmomenta_lists(dataframe)
    # Call the function for retrieving lists of 4-mom first
    output_dataframe = pd.DataFrame
    
    pi0vect = vector.obj(px = dataframe[pi0_2_4mom[1]],\
                        py = dataframe[pi0_2_4mom[2]],\
                        pz = dataframe[pi0_2_4mom[3]],\
                        E = dataframe[pi0_2_4mom[0]])

    pivect = vector.obj(px = dataframe[pi_2_4mom[1]],\
                        py = dataframe[pi_2_4mom[2]],\
                        pz = dataframe[pi_2_4mom[3]],\
                        E = dataframe[pi_2_4mom[0]])

    fourvect = vector.arr({"px": dataframe[fourmom_list_colnames[1]],\
                       "py": dataframe[fourmom_list_colnames[2]],\
                       "pz": dataframe[fourmom_list_colnames[3]],\
                        "E": dataframe[fourmom_list_colnames[0]]})
   
   
    tauvisfourvect = vector.obj(px = dataframe[tau_2_4mom[1]],\
                               py = dataframe[tau_2_4mom[2]],\
                               pz = dataframe[tau_2_4mom[3]],\
                               E = dataframe[tau_2_4mom[0]])

    pi0_phi = pi0vect.deltaphi(tauvisfourvect)
    pi0_eta = pi0vect.deltaeta(tauvisfourvect)
    pi_phi = pivect.deltaphi(tauvisfourvect)
    pi_eta = pivect.deltaphi(tauvisfourvect)

    pi0_transformed = vector.obj(px = pi0_phi, py = pi0_eta)
    pi_transformed = vector.obj(px = pi_phi, py = pi_eta)
    diff = pi_transformed - pi0_transformed

    unitvector = vector.obj(px = 1, py =0)
    rot_angles = ak.to_list(unitvector.deltaphi(diff))

    phis = ak.to_list(fourvect.deltaphi(tauvisfourvect))
    etas = ak.to_list(fourvect.deltaeta(tauvisfourvect))
    frac_energies = ak.to_list((fourvect.E/tauvisfourvect.E))
    frac_momenta = ak.to_list(fourvect.mag/tauvisfourvect.mag)

    output_dataframe = pd.DataFrame({'phis' : phis, 'etas' : etas, \
        'frac_energies' : frac_energies, 'frac_momenta' : frac_momenta, 'angles': rot_angles, "n_gammas_2": dataframe["n_gammas_2"]}) 
    # Added "n_gammas_2" column for proper parsing of photons and clusters

    return output_dataframe 

imvar_df = phi_eta_find(df_ordered)
def rot_x(x, y, theta):
    return np.array(x) * np.cos(theta) - np.array(y) * np.sin(theta)
def rot_y(x, y, theta):
    return np.array(x) * np.sin(theta) + np.array(y) * np.cos(theta)
imvar_df["rot_phi"] = imvar_df.apply(lambda x : rot_x(x["phis"], x["etas"], x["angles"]), axis = 1)
imvar_df["rot_eta"] = imvar_df.apply(lambda x : rot_y(x["phis"], x["etas"], x["angles"]), axis = 1)

cols_to_drop = ["phis", "etas", "angles"]
imvar_df.drop(columns = cols_to_drop, inplace = True)

visualise = False
if visualise:
    x = []
    y = []
    x2 = []
    y2 = []
    for index, row in imvar_df.iterrows():
        for a in range(len(row["frac_energies"])):
            if a != 0.0:
                x.append(row["rot_phi"][a])
                y.append(row["rot_eta"][a])
                x2.append(row["phis"][a])
                y2.append(row["etas"][a])
    plt.plot(x,y, 'x')
    plt.savefig('testimage.png')
    plt.plot(x2,y2, 'x')
    plt.savefig('testimage2.png')
print(imvar_df.columns)

for a in measured4mom[4:]:
    df_ordered.drop(columns = a, inplace = True)
for a in measured4mom[:4]:
    df_ordered.drop(columns = a[1:], inplace = True)
    # Keeps the energies for pi0, pi, pi2, pi3 as HL variables
for a in fourmom_list_colnames:
    df_ordered.drop(columns = a, inplace = True)
# drop irrelevant columns now they have been used for variable creation

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
print("Saving dataframes...")
time_start = time.time()

if load_full:
    pd.to_pickle(df_ordered, rootpath_save + "/ordereddf_modified.pkl")
    pd.to_pickle(imvar_df, rootpath_save + "/imvar_df.pkl")

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
