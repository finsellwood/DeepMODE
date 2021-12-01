#~~ DATAFRAMEMOD.PY ~~#
# Takes the existing n-tuple dataframe (df_ordered) and uses data to create an additional dataframe (imvar_df)
# with only the variables necessary for image creation. Then removes the columns used from df_ordered and saves
# (under diff name)

#~~ Store strings in memory ~~#
pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
pi2_2_4mom = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2", ]
pi3_2_4mom = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2", ]
gam_2_3mom = ["gam_E_2", "gam_px_2", "gam_py_2", "gam_pz_2", ]
sc1_2_4mom = ["sc1_E_2", "sc1_px_2", "sc1_py_2", "sc1_pz_2", ]
tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]
measured4mom = [pi_2_4mom, pi2_2_4mom, pi3_2_4mom, gam_2_3mom, sc1_2_4mom, ]
E_list = [a[0] for a in measured4mom]
px_list = [a[1] for a in measured4mom]
py_list = [a[2] for a in measured4mom]
pz_list = [a[3] for a in measured4mom]
fourmom_list = [E_list, px_list, py_list, pz_list]

#~~ Load in pickled dataframe ~~#
import pandas as pd
import numpy as np
import vector
import awkward as ak  
import numba as nb
import time
from sklearn.externals import joblib

df_ordered = pd.read_pickle("/vols/cms/fjo18/Masters2021/Objects/ordereddf.pkl")

#df_ordered = df_ordered.head(10000)

def phi_eta_find(dataframe):
    phiouts = ["phi_0", "phi_1", "phi_2", "phi_3", "phi_4",]
    etaouts = ["eta_0", "eta_1", "eta_2", "eta_3", "eta_4",]
    fraceouts = ["frace_0", "frace_1", "frace_2", "frace_3", "frace_4",]
    
    output_dataframe = pd.DataFrame()
    #fullmomlists(dataframe, fourmom_list)
    fourvect = vector.arr({"px": dataframe[fourmom_list[1]].values.tolist(),\
                       "py": dataframe[fourmom_list[2]].values.tolist(),\
                       "pz": dataframe[fourmom_list[3]].values.tolist(),\
                        "E": dataframe[fourmom_list[0]].values.tolist()})
   
    tauvisfourvect = vector.obj(px = dataframe[tau_2_4mom[1]],\
                               py = dataframe[tau_2_4mom[2]],\
                               pz = dataframe[tau_2_4mom[3]],\
                               E = dataframe[tau_2_4mom[0]])
    
    output_dataframe["phis"] = fourvect.deltaphi(tauvisfourvect) 
    output_dataframe["etas"] = fourvect.deltaeta(tauvisfourvect) 
    output_dataframe["frac_energies"] = fourvect.E/tauvisfourvect.E
#     output_dataframe["phis"] = [np.asarray(a, dtype = 'float32') for a in output_dataframe['phis'].values]
#     output_dataframe["etas"] = [np.asarray(a, dtype = 'float32') for a in output_dataframe['etas'].values]
#     output_dataframe["frac_energies"] = [np.asarray(a, dtype = 'float32') for a in output_dataframe['frac_energies'].values]
#     for a in range(5):
#         output_dataframe[phiouts[a]] = fourvect.deltaphi(tauvisfourvect)[a]
#         output_dataframe[etaouts[a]] = fourvect.deltaeta(tauvisfourvect)[a]
#         output_dataframe[fraceouts[a]] = (fourvect.E/tauvisfourvect.E)[a]
    #dataframe.astype({"phis": 'int32'})
    # fractional energy
    # Obsolete method ^
    return output_dataframe   


print("Finding phis and etas")
time_start = time.time()

imvar_df = phi_eta_find(df_ordered)
for a in fourmom_list:
    df_ordered.drop(columns = a, inplace = True)

time_elapsed = time_start - time.time()
print("elapsed time = " + str(time_elapsed))

pd.to_pickle(df_ordered, "/vols/cms/fjo18/Masters2021/Objects/ordereddf_modified.pkl")
joblib.dump(imvar_df, "/vols/cms/fjo18/Masters2021/Objects/imvar_df.sav")
