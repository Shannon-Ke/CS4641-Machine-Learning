# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:34:06 2018

@author: Crymsonfire
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

print(__doc__)

colors = ['r', 'g', 'b','c', 'm']
### Getting my data from local folder
nursery_data = pd.read_csv("datasets/nursery-data.csv")
car_data = pd.read_csv("datasets/cars-data.csv")

## One Hot Encoding my Data
le = LabelEncoder()
nursery = nursery_data.apply(le.fit_transform)
car = car_data.apply(le.fit_transform)

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

#iris = datasets.load_iris()
X = nursery.values[:, 0:7]
Y = nursery.values[:, 8]
#X = car.values[:, 0:5]
#Y = car.values[:, 6]
# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=5)
# Only take the first fold.
#train_index, test_index = next(iter(skf.split(iris.data, iris.target)))
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.25,
                                                        random_state = 100)
pca = PCA(0.999)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
#X_train = iris.data[train_index]
#y_train = iris.target[train_index]
#X_test = iris.data[test_index]
#y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=20, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators = len(estimators)

plt.figure(figsize=(4 * n_estimators // 2, 8))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

#    for n, color in enumerate(colors):
#        data = nursery.data[nursery.target == n]
#        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
#                    label=nursery.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


plt.show()