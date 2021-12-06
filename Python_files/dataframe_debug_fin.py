#~~ DATAFRAMEMOD.PY ~~#
# Takes the existing n-tuple dataframe (df_ordered) and uses data to create an additional dataframe (imvar_df)
# with only the variables necessary for image creation. Then removes the columns used from df_ordered and saves
# (under diff name)
rootpath = "/vols/cms/fjo18/Masters2021"

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

measured4mom = [gam_2_3mom, cl_2_3mom, ]#pi_2_4mom, pi2_2_4mom, pi3_2_4mom, sc1_2_4mom, 
E_list = [a[0] for a in measured4mom]
px_list = [a[1] for a in measured4mom]
py_list = [a[2] for a in measured4mom]
pz_list = [a[3] for a in measured4mom]
fourmom_list = [E_list, px_list, py_list, pz_list]
# a list of actual columns with the lists of fourmomenta in
fourmom_list_colnames = ["E_full_list", "px_full_list", "py_full_list", "pz_full_list"]


import pandas as pd
import numpy as np
import vector
import awkward as ak  
import numba as nb
import time
from sklearn.externals import joblib

#~~ Load in pickled dataframe ~~#

df_ordered = pd.read_pickle(rootpath + "/Objects/testhead.pkl")

print("producing new variables...")
time_start = time.time()

# for index, row in df_ordered.iterrows():
#   print(row[fourmom_list[1]].values.tolist())

# 
df_ordered["newline"] = [[]] * df_ordered.shape[0]
for index, row in df_ordered.iterrows():
    fourvect = vector.arr({"px": row[fourmom_list[1]].values.tolist(),\
                       "py": row[fourmom_list[2]].values.tolist(),\
                       "pz": row[fourmom_list[3]].values.tolist(),\
                      })
    print(df_ordered["newline"].loc[[index]])# = 
    print(fourvect.px)
time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed)) 



print(df_ordered["newline"].head())
# print(vector.arr({"px": df_ordered[fourmom_list[1]].values.tolist(),\
#                        "py": df_ordered[fourmom_list[2]].values.tolist(),\
#                        "pz": df_ordered[fourmom_list[3]].values.tolist(),\
#                       }).x)
                        #"E": df_ordered[fourmom_list[0]].values.tolist()})

#  = fourvect.px
#df_ordered["newline"].head()
# def energyfinder(dataframe, momvariablenames_1):
#     arr = []
#     counter = -1
#     for i in dataframe[momvariablenames_1[1]]:
#         counter += 1
#         energ = []
#         if i.size > 0:
#             for j in range(len(i)):
#                 momvect_j = [dataframe[momvariablenames_1[1]][counter][j],\
#                              dataframe[momvariablenames_1[2]][counter][j],\
#                              dataframe[momvariablenames_1[3]][counter][j]]
#                 energ.append(np.sqrt(sum(x**2 for x in momvect_j)))
#             arr.append(energ)
#         else: arr.append([])
#     dataframe[momvariablenames_1[0]] = arr

# This function generates the energy of the given massless object (magnitude of 3-mom) 
# So that can be treated the same as the other four-momenta later

# energyfinder(df_ordered, gam_2_3mom)
# energyfinder(df_ordered, cl_2_3mom)
# define the energy variables for the photon and cluster lists


# def inv_mass(Energ,Px,Py,Pz):
#     vect = vector.obj(px=Px, py=Py, pz=Pz, E=Energ)
#     return vect.mass

# df_ordered["pi0_2mass"] = inv_mass(df_ordered["pi0_E_2"],df_ordered["pi0_px_2"],df_ordered["pi0_py_2"],df_ordered["pi0_pz_2"]) #pion masses

# def rho_mass(dataframe, momvariablenames_1, momvariablenames_2):
#     momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
#                        py = dataframe[momvariablenames_1[2]],\
#                        pz = dataframe[momvariablenames_1[3]],\
#                        E = dataframe[momvariablenames_1[0]])
#     momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
#                        py = dataframe[momvariablenames_2[2]],\
#                        pz = dataframe[momvariablenames_2[3]],\
#                        E = dataframe[momvariablenames_2[0]])
#     rho_vect = momvect1+momvect2
#     dataframe["rho_mass"] = inv_mass(rho_vect.E, rho_vect.px, rho_vect.py, rho_vect.pz) #rho masses
    
# rho_mass(df_ordered, pi_2_4mom, pi0_2_4mom)
# # rho mass is the addition of the four-momenta of the charged and neutral pions

# df_ordered["E_gam/E_tau"] = df_ordered["gam1_E_2"].divide(df_ordered["tau_E_2"]) #Egamma/Etau
# df_ordered["E_pi/E_tau"] = df_ordered["pi_E_2"].divide(df_ordered["tau_E_2"]) #Epi/Etau
# df_ordered["E_pi0/E_tau"] = df_ordered["pi0_E_2"].divide(df_ordered["tau_E_2"]) #Epi0/Etau

# def tau_eta(dataframe, momvariablenames_1):
#     momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
#                        py = dataframe[momvariablenames_1[2]],\
#                        pz = dataframe[momvariablenames_1[3]],\
#                        E = dataframe[momvariablenames_1[0]])
#     dataframe["tau_eta"] = momvect1.eta  #tau eta (tau pt just a variable)
    
# tau_eta(df_ordered, tau_2_4mom)

# def ang_var(dataframe, momvariablenames_1, momvariablenames_2, particlename): #same for gammas and pions
#     momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
#                        py = dataframe[momvariablenames_1[2]],\
#                        pz = dataframe[momvariablenames_1[3]],\
#                        E = dataframe[momvariablenames_1[0]])
#     momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
#                        py = dataframe[momvariablenames_2[2]],\
#                        pz = dataframe[momvariablenames_2[3]],\
#                        E = dataframe[momvariablenames_2[0]])
    
#     diffphi = momvect1.phi - momvect2.phi
#     diffeta = momvect1.eta - momvect2.eta
#     diffr = np.sqrt(diffphi**2 + diffeta**2)
#     Esum = dataframe[momvariablenames_1[0]] + dataframe[momvariablenames_2[0]]
#     dataframe["delR_"+ particlename] = diffr
#     dataframe["delPhi_"+ particlename] = diffphi
#     dataframe["delEta_" + particlename] = diffeta
#     dataframe["delR_xE_"+ particlename] = diffr * Esum
#     dataframe["delPhi_xE_"+ particlename] = diffphi * Esum
#     dataframe["delEta_xE_" + particlename] = diffeta * Esum
        
# ang_var(df_ordered, gam1_2_4mom, gam2_2_4mom, "gam")
# ang_var(df_ordered, pi0_2_4mom, pi_2_4mom, "pi")

# time_elapsed = time_start - time.time()
# print("elapsed time = " + str(time_elapsed))
# print("Finding phis and etas")
# time_start = time.time()

# def phi_eta_find(dataframe):  
#     output_dataframe = pd.DataFrame()

#     fourvect = vector.arr({"px": dataframe[fourmom_list[1]].values.tolist(),\
#                        "py": dataframe[fourmom_list[2]].values.tolist(),\
#                        "pz": dataframe[fourmom_list[3]].values.tolist(),\
#                         "E": dataframe[fourmom_list[0]].values.tolist()})
   
#     tauvisfourvect = vector.obj(px = dataframe[tau_2_4mom[1]],\
#                                py = dataframe[tau_2_4mom[2]],\
#                                pz = dataframe[tau_2_4mom[3]],\
#                                E = dataframe[tau_2_4mom[0]])
    
#     output_dataframe["phis"] = fourvect.deltaphi(tauvisfourvect) 
#     output_dataframe["etas"] = fourvect.deltaeta(tauvisfourvect) 
#     output_dataframe["frac_energies"] = fourvect.E/tauvisfourvect.E
    
#     return output_dataframe   

# imvar_df = phi_eta_find(df_ordered)

# for a in fourmom_list:
#     df_ordered.drop(columns = a, inplace = True)

# time_elapsed = time_start - time.time()
# print("elapsed time = " + str(time_elapsed))

# pd.to_pickle(df_ordered, rootpath + "/Objects/ordereddf_modified.pkl")
# joblib.dump(imvar_df, rootpath + "/Objects/imvar_df.sav")
