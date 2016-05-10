import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cPickle
import Bdata


bald = Bdata.load_data()
clf = svm.SVC(gamma=0.0001, C=100)
x, y = bald.data[:], bald.target[:]
clf.fit(x, y)

with open('learnData1930', 'wb') as f:
    pick = cPickle.dump(clf, f)