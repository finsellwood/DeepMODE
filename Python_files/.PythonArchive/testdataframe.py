import pandas as pd
df = pd.read_pickle("/vols/cms/fjo18/Masters2021/imvar_df.pkl")
#df2 = pd.read_pickle("/vols/cms/fjo18/Masters2021/imvar_df_old.pkl")
# ordereddf_modified
for a in df.columns.values.tolist():
   print(df[a].head())#, type(df[a][0][0]))
print(df.columns.values.tolist())
  #len(df[a][0]), 
# for index, row in df.head().iterrows():
#     print('this is row', index)
#     for a in row:
#         print(a)
    
