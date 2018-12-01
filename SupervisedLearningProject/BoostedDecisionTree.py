import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
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
car_x = []
car_y = []
testx = []
testy = []

for i in range(1,10):
    ## Splitting up features and target for nursery
     # and making training and test data sets
    X = nursery.values[:, 0:7]
    Y = nursery.values[:, 8]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3,
                                                        random_state = 100)
    ## Splitting up features and target for car
     # and making training and test data sets
    W = car.values[:, 0:5]
    Z = car.values[:, 6]
    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.1,
                                                        random_state = 100)
#    clf = GradientBoostingClassifier(n_estimators=i * 3, learning_rate=1.0,
#                                     max_depth=1, random_state=100).fit(X_train, y_train)
    
    ## Running my car data into the decision tree algorithm using information gain
    
    
    ## predicting accuracy
    
#    print "Accuracy for nursery data is ", clf.score(X_test, y_test)*100
#    nursery_x.append(i*3)
#    nursery_y.append(clf.score(X_test, y_test)*100)
    clf2 = GradientBoostingClassifier(n_estimators=i * 3, learning_rate=1.0,
                                     max_depth=2, random_state=100)
    scores = cross_val_score(clf2, W_train, z_train, cv=5)
    clf2.fit(W_train, z_train)
    testx.append(i*3)
    testy.append(scores.mean()*100)
    num = clf2.score(W_train, z_train)*100
    print "Accuracy for car data is ", num
    car_x.append(i*3)
    car_y.append(num)
#plt.plot(nursery_x,nursery_y, label="Training")
plt.plot(car_x,car_y, label="Training")
plt.plot(testx, testy, label="Cross Validation")

plt.legend(loc='upper left', frameon=False)
plt.xlabel("Number of Boosting Stages")
plt.ylabel("Accuracy (%)")
plt.title("Car - Boosted Decision Trees")