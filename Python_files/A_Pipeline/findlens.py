###find lengths###
import pandas as pd
rootpath = "/vols/cms/fjo18/Masters2021/A_Objects/Objects3"

df_mod_names = ["/df_m_dm0.pkl", "/df_m_dm1.pkl", "/df_m_dm2.pkl", \
    "/df_m_dm3.pkl", "/df_m_dm4.pkl", "/df_m_dm5.pkl"]
for a in df_mod_names:
    x = pd.read_pickle(rootpath+a)
    print(x.shape)