import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
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

for i in range(1,20):
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
    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = (100 - i)/100.0,
                                                        random_state = 100)
    
    knn = KNeighborsClassifier(n_neighbors=i*10)    
    #Train the model using the training sets
    knn.fit(X_train, y_train)    
    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    testx.append(i*10)
    testy.append(scores.mean()*100)
    print "Accuracy for nursery is :",accuracy_score(y_test, y_pred)*100
    nursery_x.append(i*10)
    nursery_y.append(accuracy_score(y_test,y_pred)*100)
    
#    knn2 = KNeighborsClassifier(n_neighbors=5)
#    
#    #Train the model using the training sets
#    knn2.fit(W_train, z_train)
#    
#    #Predict the response for test dataset
#    z_pred = knn2.predict(W_test)
#    
#    print "Accuracy for cars is :",accuracy_score(z_test, z_pred)*100
#    car_x.append(i/100.0)
#    car_y.append(accuracy_score(z_test,z_pred)*100)
plt.plot(nursery_x,nursery_y, label="Training")
plt.plot(testx, testy, label="Cross Validation")
#plt.plot(car_x,car_y, label="car")
plt.legend(loc='upper left', frameon=False)
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy (%)")
plt.title("Nursery - kNN")
    
    
