import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

### Getting my data from local folder
nursery_data = pd.read_csv("datasets/nursery-data.csv")
car_data = pd.read_csv("datasets/cars-data.csv")

## One Hot Encoding my Data
le = LabelEncoder()
nursery = nursery_data.apply(le.fit_transform)
car = car_data.apply(le.fit_transform)

nursery_x = []
nursery_y = []
nursery_test = []
nursery_testy = []
car_x = []
car_y = []

for i in range(1,12):
    
    ## Splitting up features and target for nursery
     # and making training and test data sets
    X = nursery.values[:, 0:7]
    Y = nursery.values[:, 8]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3,
                                                        random_state = 100)
#    
    ## Splitting up features and target for car
     # and making training and test data sets
    W = car.values[:, 0:5]
    Z = car.values[:, 6]
    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.1,
                                                        random_state = 100)
    
    ## Running my nursery data into the decision tree algorithm using information gain
    clf_entropy = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=i,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=100, splitter='best')
    
   
    
    ## Running my car data into the decision tree algorithm using information gain
#    clf_entropy2 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=i,
#                max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#                min_samples_split=2, min_weight_fraction_leaf=0.0,
#                presort=False, random_state=100, splitter='best')
#    clf_entropy2.fit(W_train, z_train)
#    
    scores = cross_val_score(clf_entropy, W_train, z_train, cv=5)
    nursery_test.append(i)
    nursery_testy.append(scores.mean()*100)
    clf_entropy.fit(W_train, z_train)
    ## predicting accuracy
    z_pred_en = clf_entropy.predict(W_test)
    print "Accuracy for nursery data is ", accuracy_score(z_test,z_pred_en)*100
    nursery_x.append(i)
    nursery_y.append(accuracy_score(z_test,z_pred_en)*100)
    
#    z_pred_en = clf_entropy2.predict(W_test)
#    print "Accuracy for car data is ", accuracy_score(z_test,z_pred_en)*100
#    car_x.append(i)
#    car_y.append(accuracy_score(z_test,z_pred_en)*100)
plt.plot(nursery_x,nursery_y, label="Training")
plt.plot(nursery_test,nursery_testy, label="Cross Validation")
plt.legend(loc='lower right', frameon=False)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy (%)")
plt.title("Car - Decision Trees")