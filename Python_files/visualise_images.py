#~~ VISUALISE_IMAGES.py ~~#
# This file loads in the numpy image arrays to ensure they're working properly
# Saves a figure with 20 images on (large + small side-by-side) for examination
rootpath = "/vols/cms/fjo18/Masters2021"
num_arrays = 73


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#~~ Load pkl files ~~#
###
time_start = time.time()
print("loading in large image arrays...")
###

list_of_arrays = []
for a in range(num_arrays):
  list_of_arrays.append(np.load(rootpath + "/Images/image_l_%02d.npy" % a))
l_image_array = np.concatenate(list_of_arrays)
list_of_arrays = []
print('large arrays shape' + l_image_array.shape)

###
time_elapsed = time.time() - time_start
time_start = time.time()
print("elapsed time = " + str(time_elapsed))
print("loading in small image arrays...")
###

for a in range(num_arrays):
  list_of_arrays.append(np.load(rootpath + "/Images/image_s_%02d.npy" % a))
s_image_array = np.concatenate(list_of_arrays)
list_of_arrays = []
print('small arrays shape' + s_image_array.shape)


import random
no_images = 10
no_arrays = s_image_array.shape[0]
fig, ax = plt.subplots(no_images,2)
fig.set_size_inches(12, 40)
for a in range(no_images):
    g = random.randint(0,no_arrays)

    ax[a][0].imshow(l_image_array[g], cmap='gray_r')
    ax[a][1].imshow(s_image_array[g], cmap='gray_r')
    
plt.savefig('images.png', dpi = 500)
