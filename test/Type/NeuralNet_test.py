from __future__ import print_function

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import Bdata

# Load Data
bald = Bdata.load_data()

x, y = bald.data[:], bald.target[:]

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

classifier.fit(x, y)

print()
print (classifier.predict(bald.data[-5]))
print (classifier.predict(bald.data[-4]))
print (classifier.predict(bald.data[-3]))
print (classifier.predict(bald.data[-2]))
print (classifier.predict(bald.data[-1]))


h = .03

reduced_data = PCA(n_components=2).fit_transform(bald.data)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]).fit(reduced_data[:, :], y)

x_min, x_max = reduced_data[:, 0].min() - 5, reduced_data[:, 0].max() + 5
y_min, y_max = reduced_data[:, 1].min() - 5, reduced_data[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = classifier.predict((np.c_[xx.ravel(), yy.ravel()]))

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