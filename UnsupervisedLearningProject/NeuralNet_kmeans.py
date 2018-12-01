# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:54:51 2018

@author: Crymsonfire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import csv

### Getting my data from local folder
car_data = pd.read_csv("datasets/cars-data.csv")
## One Hot Encoding my Data
le = LabelEncoder()
car = car_data.apply(le.fit_transform)
car_x = []
car_y = []
W = car.values[:, 0:6]
Z = car.values[:, 6]
W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.25,
                                                            random_state = 100)
kmeans = KMeans(n_clusters=4) 
kmeans.fit(W_train)
kmeans_predict1 = kmeans.predict(W_train)
#with open('datasets/cars-data.csv','r') as csvinput:
#    with open('datasets/cars-data-kmeans.csv', 'w') as csvoutput:
#        writer = csv.writer(csvoutput, lineterminator='\n')
#        reader = csv.reader(csvinput)
#
#        all = []
#        row = next(reader)
#        row.append(kmeans_predict1[i])
#        all.append(row)
#
#        for row in reader:
#            row.append(row[0])
#            all.append(row)
#
#        writer.writerows(all)
        
avg = 0.0
numcalled = 0

y = []
car_data = pd.read_csv("datasets/cars-data-kmeans.csv")
    ## One Hot Encoding my Data

car = car_data.apply(le.fit_transform)
for i in range (1,14):
    numcalled += 1
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
#    
    
#    W_train = kmeans.transform(W_train)
#    W_test = kmeans.transform(W_test)
##    
#    print(W_train)

#    for i in range(len(W)):
#        W[i].append(kmeans_predict1[i])
    
    W = car.values[:, 0:7]
    Z = car.values[:, 7]
    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.25,
                                                            random_state = 100)
    
#    gmm = GaussianMixture(n_components=4,
#                   covariance_type='tied', max_iter=20, random_state=0)
#    gmm.fit(W_train)
#    print(gmm.predict_proba(W_train))
    
    mlp.fit(W_train,z_train)
    
    prediction = mlp.predict(W_train)
#    print(accuracy_score(z_train, prediction)*100)
    predictions = mlp.predict(W_test)
    car_x.append(numcalled)
    y.append(accuracy_score(z_train, prediction)*100)
    car_y.append(accuracy_score(z_test,predictions)*100)  
    #print(accuracy_score(z_test,predictions)*100) 
#    print(confusion_matrix(z_test,predictions))
#    print(classification_report(z_test,predictions))
    avg += accuracy_score(z_test,predictions)*100
print(avg/numcalled)
 
#plt.plot(nursery_x,nursery_y, label="nursery")
plt.figure(figsize = (8,6))
plt.plot(car_x,car_y, label="testing accuracy")
plt.plot(car_x, y, label="training accuracy")
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', frameon=False)
plt.title('Kmeans with Cars Data on a NN')