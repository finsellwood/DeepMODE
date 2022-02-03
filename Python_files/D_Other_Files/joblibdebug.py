#~~ joblibdebug.PY ~~#
# Converts joblib .save file to pickle file so can vbe used more easily
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
