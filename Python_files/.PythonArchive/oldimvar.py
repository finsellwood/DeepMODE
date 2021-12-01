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

df_ordered = pd.read_pickle("/vols/cms/fjo18/Masters2021/ordereddf.pkl")

df_ordered = df_ordered.head(1000)

# def fullmomlists(dataframe):
#     dataframe["E_full_list"] = dataframe[fourmom_list[0]].values.tolist()
#     dataframe["px_full_list"] = dataframe[fourmom_list[1]].values.tolist()
#     dataframe["py_full_list"] = dataframe[fourmom_list[2]].values.tolist()
#     dataframe["pz_full_list"] = dataframe[fourmom_list[3]].values.tolist()
# This command is now obsolete with the updated phi_eta_find

def phi_eta_find(dataframe):
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
    #dataframe.astype({"phis": 'int32'})
    # fractional energy
    return output_dataframe



def largegrid(dataframe, imvarfourmomentum_cols, dimension_l, dimension_s):
    phi_eta_find(dataframe,output_dataframe)
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
    


#fullmomlists(df_ordered,fourmom_list)

print("Finding phis and etas")
time_start = time.time()

imvar_df = phi_eta_find(df_ordered)
for a in fourmom_list:
    df_ordered.drop(columns = a, inplace = True)
    
# (TESTING AREA TO SOLVE MEMORY ERRORS)

time_elapsed = time_start - time.time()
print("elapsed time = " + str(time_elapsed))

pd.to_pickle(df_ordered, "/vols/cms/fjo18/Masters2021/ordereddf_modified.pkl")
pd.to_pickle(imvar_df, "/vols/cms/fjo18/Masters2021/imvar_df_old.pkl")
