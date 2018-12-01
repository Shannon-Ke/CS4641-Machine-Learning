# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 11:36:22 2018

@author: Crymsonfire
"""

print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition
from sklearn.decomposition import PCA

np.random.seed(5)

### Getting my data from local folder
nursery_data = pd.read_csv("datasets/nursery-data.csv")
car_data = pd.read_csv("datasets/cars-data.csv")

## One Hot Encoding my Data
le = LabelEncoder()
nursery = nursery_data.apply(le.fit_transform)
car = car_data.apply(le.fit_transform)
feature = ['parents', 'has_nurs', 'form' ,' children' ,'housing', 'finance'
            , 'social', 'health']
X = nursery.values[:, 0:7]

X = StandardScaler().fit_transform(X)
#Y = nursery.values[:, 8]
Y = nursery_data.loc[:,['label']].values
features=['buying', 'maint', 'doors', 'persons', 'lug_boot' ,'safety']
W = car.loc[:, features].values

W = StandardScaler().fit_transform(W)
#Z = car.values[:, 6]
Z = car_data.loc[:,['label']].values

pca = PCA(n_components = 2) #change between 2 and 3

principalComponents = pca.fit_transform(W)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf)
finalDf = pd.concat([principalDf, car_data[['label']]], axis = 1)
print(finalDf)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['acc', 'good', 'unacc', 'vgood']
colors = ['r', 'g', 'b', 'c']
#targets = ['not_recom', 'priority', 'recommend', 'spec_prior', 'very_recom']
#colors = ['r', 'g', 'b','c', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


print("******PCA EXPLAINED******", pca.explained_variance_ratio_)


#centers = [[1, 1], [-1, -1], [1, -1]]
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

#fig = plt.figure(1, figsize=(4, 3))
#plt.clf()
#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
#plt.cla()
#pca = decomposition.PCA(n_components=5)
#pca.fit(X)
#principalComponents = pca.fit_transform(X)
#X = pca.transform(X)
##principalDf = pd.DataFrame(data = principalComponents
##             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#for name, label in [('recommend', 2), ('priorty', 1), ('not_recom', 0), ('spec_prior', 3), ('very_recom', 4)]:
#    ax.text3D(X[Y == label, 0].mean(),
#              X[Y == label, 1].mean() + 1.5,
#              X[Y == label, 2].mean(),
#           
#              name,
#              horizontalalignment='center',
#              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
## Reorder the labels to have colors matching the cluster results
#Y = np.choose(Y, [4, 3, 1, 2, 0]).astype(np.float)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.nipy_spectral,
#           edgecolor='k')
#
#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])
#
#plt.show()