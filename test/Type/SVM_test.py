import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

import Bdata

bald = Bdata.load_data()
# bald = test_data.load_data()

clf = svm.SVC(gamma=0.0001, C=100)

print (len(bald.data))

x, y = bald.data[:], bald.target[:]

clf.fit(x, y)

print clf.predict(bald.data[-5])
print clf.predict(bald.data[-4])
print clf.predict(bald.data[-3])
print clf.predict(bald.data[-2])
print clf.predict(bald.data[-1])

h = .03

reduced_data = PCA(n_components=2).fit_transform(bald.data)

svc = svm.SVC(gamma=0.0001, C=100, kernel='rbf').fit(reduced_data[:, :], y)

x_min, x_max = reduced_data[:, 0].min() - 5, reduced_data[:, 0].max() + 5
y_min, y_max = reduced_data[:, 1].min() - 5, reduced_data[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict((np.c_[xx.ravel(), yy.ravel()]))

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()