# DeepMODE
Repository of code for applying machine learning methods to CMS events, with the hope of improving tau decay mode resolution.

1) install_script.sh
installs the necessary packages 
2) dataframe_init.sh
Loads the dataframes from root files, chooses relevant columns and saves as .pkl file - does not create new variables.
3) dataframe_mod.sh
Unpacks .pkl dataframe (df_ordered) and creates new variables, and saves dataframe. Also creates separate dataframe (imvar_df) for variables for creating images, which is saved as imvar_df.sav (using joblib to avoid memory errors).
4) image_generator.sh implements imgen.py, to generate and save numpy arrays of images in 100,000 event batches - images are compressed by using uint8 format rather than f32. 
5) dataframe_split.sh separates the data for HL vars, large images, small images and y values into training and test data, saves in tf.data.dataset format, which can be efficiently cycled in and out of ram to reduce memory usage.
6) train_model.sh loads tensors from /vols/cms/fjo18/Masters2021/Tensors and trains a model on them, before saving the model back into /vols/. Note: only saves model if 'save_model' parameter is set to 'True' in the 'train_NN.py' file
7) train_model_HL.sh trains only on the HL variables (using train_HLNN.py file). Same parameters options as for full model.
