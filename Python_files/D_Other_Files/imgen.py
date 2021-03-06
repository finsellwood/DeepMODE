#~~ IMGEN.PY ~~#
# Takes imvar_df dataframe and generates image files in batches of 100,000 events. Saves as numpy arrays.
rootpath = "/vols/cms/fjo18/Masters2021"
debug  = False
#~~ Packages ~~#
import pandas as pd
import numpy as np
import vector
import awkward as ak  
import numba as nb
import time
from sklearn.externals import joblib
import pylab as pl

#~~ Load the dataframe with image variables in ~~#
imvar_df = pd.read_pickle(rootpath + "/Objects/imvar_df.pkl")
if debug:
    imvar_df = imvar_df.head(1000)


#~~ Function to generate images ~~#
def largegrid(dataframe, dimension_l, dimension_s):
    halfdim = dimension_l/2
    halfdim2 = dimension_s/2
    largegridlist = []
    smallgridlist = []
    imcounter = 0
    counter = 0
    for index, row in dataframe.iterrows():
        grid = np.zeros((dimension_l,dimension_l), np.uint8)
        grid2 = np.zeros((dimension_s,dimension_s), np.uint8)
        phis = np.array(row["phis"])
        etas = np.array(row["etas"])
        energies = row["frac_energies"]

        # ARRAY SIZES: outer is 21x21, -0.6 to 0.6 in phi/eta
        #              inner is 11x11, -0.1 to 0.1 in phi/eta


        phicoords =  [max(min(a, 20), 0) for a in np.floor((phis/1.21) * dimension_l + halfdim).astype(int)]
        etacoords =  [max(min(a, 20), 0) for a in np.floor(-1 * (etas/1.21) * dimension_l + halfdim).astype(int)]
        phicoords2 = [max(min(a, 10), 0) for a in np.floor((phis/0.2) * dimension_s + halfdim2).astype(int)]
        etacoords2 = [max(min(a, 10), 0) for a in np.floor(-1 * (etas/0.2) * dimension_s + halfdim2).astype(int)]
        for a in range(len(energies)):
            # if energies[a] != 0.0:
                
            grid[etacoords[a]][phicoords[a]] += int(min(abs(energies[a]), 1) * 255)
                # NOTE - if sum of elements exceeds 255 for a given cell then it will loop back to zero
            #if etacoords2[a] < dimension_s and etacoords2[a] >= 0 and phicoords2[a] < dimension_s and phicoords2[a] >=0:
            grid2[etacoords2[a]][phicoords2[a]] += int(min(abs(energies[a]), 1) * 255)
            
        largegridlist.append(grid)
        smallgridlist.append(grid2)
        
        counter +=1
        if counter == 100000:
            np.save(rootpath + '/Images/image_l_%02d.npy' % imcounter, largegridlist)
            np.save(rootpath + '/Images/image_s_%02d.npy' % imcounter, smallgridlist)
            largegridlist = []
            smallgridlist = []
            print('Images saved = ', imcounter)
            imcounter+=1
            counter = 0
    np.save(rootpath + '/Images/image_l_%02d.npy' % imcounter, largegridlist)
    np.save(rootpath + '/Images/image_s_%02d.npy' % imcounter, smallgridlist)


     

largegrid(imvar_df, 21, 11)
