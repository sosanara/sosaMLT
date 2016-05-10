from sklearn.neighbors import RadiusNeighborsClassifier

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.decomposition import PCA

radius = 1.0

import Bdata

bald = Bdata.load_data()

x, y = bald.data[:], bald.target[:]


clf = RadiusNeighborsClassifier(radius=radius)
clf.fit(x, y)


print clf.predict(bald.data[-5])
print clf.predict(bald.data[-4])
print clf.predict(bald.data[-3])
print clf.predict(bald.data[-2])
print clf.predict(bald.data[-1])
