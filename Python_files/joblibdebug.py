#~~ IMGEN.PY ~~#
# Takes imvar_df dataframe and generates image files in batches of 100,000 events. Saves as numpy arrays.
rootpath = "/vols/cms/fjo18/Masters2021"
#~~ Packages ~~#
import pandas as pd
import numpy as np
import vector
import awkward as ak  
import numba as nb
import time
from sklearn.externals import joblib

#~~ Load the dataframe with image variables in ~~#
imvar_df = joblib.load(rootpath + "/Objects/imvar_df.sav")
pd.to_pickle(imvar_df, rootpath + "/Objects/imvar_df_debug.pkl")
