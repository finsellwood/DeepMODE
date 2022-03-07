import matplotlib.pyplot as pl
import numpy as np
data = np.array([1271940, 2534458, 1122514, 1265396, 635299, 576278, ])
modes = [0,1,2,3,4,5]
pl.bar(modes, data)
pl.show()
pl.savefig("jostpgram.png")
print(sum(data)/60000)