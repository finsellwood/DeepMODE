#~~ IMGEN.PY ~~#
# Takes imvar_df dataframe and generates image files in batches of 100,000 events. Saves as numpy arrays.
# IMGEN_MULTI - adds multiple layers rather than just one for energy
rootpath = "/vols/cms/fjo18/Masters2021"
debug  = False
large_image_dim = 21
small_image_dim = 11
no_layers = 4

#~~ Packages ~~#
import pandas as pd
import numpy as np
import vector
import awkward as ak  
import numba as nb
import time
#from sklearn.externals import joblib
import pylab as pl


#~~ Load the dataframe with image variables in ~~#

print("Loading dataframes...")
time_start = time.time()

if debug:
    imvar_df = pd.read_pickle(rootpath + "/Objects/imvar_df_debug.pkl")
else:
    imvar_df = pd.read_pickle(rootpath + "/Objects/imvar_df.pkl")
print("loaded")

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
time_start = time.time()

pi_indices = np.array([0,1,1,1])
pi0_indices = np.array([1,0,0,0,])
gamma_indices = np.array([0,0,0,0,])
# pi0, pi1, pi2, pi3, sc1, gam_list, cluster_list 
# Should be enough zeros to account for the last bulk of the coordinates, which are neutral photons etc
#~~ Function to generate images ~~#
def largegrid(dataframe, dimension_l, dimension_s, no_layers):
    no_layers = no_layers
    halfdim = dimension_l/2
    halfdim2 = dimension_s/2
    largegridlist = []
    smallgridlist = []
    imcounter = 0
    counter = 0
    fulllen = dataframe.shape[0]
    for index, row in dataframe.iterrows():
        print(index/fulllen)
        grid = np.zeros((dimension_l,dimension_l, no_layers), np.uint8)
        grid2 = np.zeros((dimension_s,dimension_s, no_layers), np.uint8)
        # Two image grids (two vals per cell, [0] for energy and [1] for charge)
        phis = np.array(row["phis"])
        etas = np.array(row["etas"])
        energies = row["frac_energies"]

        # ARRAY SIZES: outer is 21x21, -0.6 to 0.6 in phi/eta
        #              inner is 11x11, -0.1 to 0.1 in phi/eta


        phicoords =  [max(min(a, 20), 0) for a in np.floor((phis/1.21) * dimension_l + halfdim).astype(int)]
        etacoords =  [max(min(a, 20), 0) for a in np.floor(-1 * (etas/1.21) * dimension_l + halfdim).astype(int)]
        phicoords2 = [max(min(a, 10), 0) for a in np.floor((phis/0.2) * dimension_s + halfdim2).astype(int)]
        etacoords2 = [max(min(a, 10), 0) for a in np.floor(-1 * (etas/0.2) * dimension_s + halfdim2).astype(int)]

        int_energies = (np.minimum(np.abs(energies), 1) * 255).astype(np.uint8)
        real_mask = int_energies != 0
        masklen = len(real_mask)
        #print(real_mask)
        # Create a mask to remove imaginary particles
        zerobuffer = np.zeros(masklen-4)
        onebuffer = np.ones(masklen-4)
        pi_count = np.append(pi_indices, zerobuffer)
        pi0_count = np.append(pi0_indices, zerobuffer)
        gamma_count = np.append(gamma_indices, onebuffer)
        layerlist = [int_energies, pi_count*real_mask, pi0_count*real_mask, gamma_count*real_mask]

        for a in range(len(energies)):
            # if energies[a] != 0.0:
            for b in range(no_layers):
                grid[etacoords[a],phicoords[a], b] += layerlist[b][a]
                # NOTE - if sum of elements exceeds 255 for a given cell then it will loop back to zero
                #if etacoords2[a] < dimension_s and etacoords2[a] >= 0 and phicoords2[a] < dimension_s and phicoords2[a] >=0:
                grid2[etacoords2[a],phicoords2[a], b] += layerlist[b][a]
                # Iterates through no_layers, so each layer has properties based on related layerlist component
        largegridlist.append(grid)
        smallgridlist.append(grid2)
        
        counter +=1
        if counter == 100000:
            np.save(rootpath + '/Images/m_image_l_%02d.npy' % imcounter, largegridlist)
            np.save(rootpath + '/Images/m_image_s_%02d.npy' % imcounter, smallgridlist)
            largegridlist = []
            smallgridlist = []
            print('Images saved = ', imcounter)
            imcounter+=1
            counter = 0
    np.save(rootpath + '/Images/m_image_l_%02d.npy' % imcounter, largegridlist)
    np.save(rootpath + '/Images/m_image_s_%02d.npy' % imcounter, smallgridlist)



largegrid(imvar_df, large_image_dim, small_image_dim, 4)

time_elapsed = time.time() - time_start
print("elapsed time = " + str(time_elapsed))
time_start = time.time()