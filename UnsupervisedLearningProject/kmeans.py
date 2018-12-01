# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:49:13 2018

@author: Crymsonfire
"""

# Dependencies
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

### Getting my data from local folder
nursery_data = pd.read_csv("datasets/nursery-data.csv")
car_data = pd.read_csv("datasets/cars-data.csv")

## One Hot Encoding my Data
le = LabelEncoder()
nursery = nursery_data.apply(le.fit_transform)
car = car_data.apply(le.fit_transform)

## Splitting up features and target for nursery
 # and making training and test data sets
X = nursery.values[:, 0:7]
Y = nursery.values[:, 8]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3,
                                                    random_state = 100)

#print(nursery.info())
#    
## Splitting up features and target for car
 # and making training and test data sets
W = car.values[:, 0:5]
Z = car.values[:, 6]
W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.2,
                                                    random_state = 100)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
W_scaled = scaler.fit_transform(W_train)

x = []
y = []
correct = 0

#pca = PCA(.70)
#pca.fit(X_train)
#
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)
kmeans = KMeans(n_clusters=5) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X_train)

print("training", len(X_train))
for i in range(len(X_train)):
    predict_me = X_train[i]
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
#    print("prediction ", prediction[0], y_train[i])
    x.append(prediction[0]);
    y.append(y_train[i])
    if prediction[0] == y_train[i]:
        correct += 1

print("num correct" ,correct)
print(correct/(float)(len(X_train)))

n, bins, patches = plt.hist(x, bins='auto', color='blue',
                            alpha=0.7, rwidth=0.99)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.title('KMeans Clustering on Nursery Dataset')
plt.text(23, 45, r'$\mu=15, b=3$')

#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#
## We can set the number of bins with the `bins` kwarg
#axs[0].hist(x, bins=20)
#axs[1].hist(y, bins=20)






w = []
z = []
W_scaled = scaler.fit_transform(W_train)

pca = PCA(.70)
pca.fit(W_train)

W_train = pca.transform(W_train)
W_test = pca.transform(W_test)

kmeans2 = KMeans( n_clusters=4) # You want cluster the passenger records into 2: Survived or Not survived
kmeans2.fit(W_train)
correct2 = 0
print("training", len(W_train))
pred = []
for i in range(len(W_train)):
    predict_me = W_train[i]
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans2.predict(predict_me)
   # print("prediction ", prediction[0], z_train[i])
    pred.append(prediction[0])
    w.append(prediction[0]);
    z.append(z_train[i])
    if prediction[0] == z_train[i]:
        correct2 += 1
print("num correct" ,correct2)
print(correct2/(float)(len(W_train)))
#
#n, bins, patches = plt.hist(w, bins='auto', color='green',
#                            alpha=0.7, rwidth=0.99)
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Cluster')
#plt.ylabel('Number of Samples')
#plt.title('KMeans Clustering on Cars Dataset')
#plt.text(23, 45, r'$\mu=15, b=3$')