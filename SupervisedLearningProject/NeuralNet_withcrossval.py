import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
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
layerNum = ()
testx = []
testy = []
testcx = []
testcy = []

for i in range(1,10):
    ## Splitting up features and target for nursery
     # and making training and test data sets
    X = nursery.values[:, 0:7]
    Y = nursery.values[:, 8]
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3,
                                                        random_state = 0)
    ## Splitting up features and target for car
     # and making training and test data sets
    W = car.values[:, 0:5]
    Z = car.values[:, 6]
    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.1,
                                                        random_state = 0)
    
    scaler = StandardScaler()
#    scaler.fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    
#    mlp = MLPClassifier(hidden_layer_sizes=layerNum + (30,))
#    mlp.fit(X_train,y_train)
#    
#    predictions = mlp.predict(X_test)
#    #print(confusion_matrix(y_test,predictions))
#    #print(classification_report(y_test,predictions))
#    print "Accuracy for nurse: ", accuracy_score(y_test,predictions)*100
    ##
#    nursery_x.append(i)
#    nursery_y.append(accuracy_score(y_test,predictions)*100)
#    scores = cross_val_score(mlp, X_train, y_train, cv=5)
#    testx.append(i)
#    testy.append(scores.mean()*100)
    
    scaler.fit(W_train)
    W_train = scaler.transform(W_train)
    W_test = scaler.transform(W_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=layerNum + (60,))
    mlp.fit(W_train,z_train)
    
    predictions = mlp.predict(W_test)
    car_x.append(i)
    car_y.append(accuracy_score(z_test,predictions)*100)
    scorescar = cross_val_score(mlp, W_train, z_train, cv=5)
    testcx.append(i)
    testcy.append(scorescar.mean()*100)
    layerNum += (60,)
    
    #print(confusion_matrix(z_test,predictions))
    #print(classification_report(z_test,predictions))
#plt.plot(nursery_x,nursery_y, label="Training")
#plt.plot(testx, testy, label="Cross Validation")
#
#plt.legend(loc='upper left', frameon=False)
#
#plt.xlabel("Number of Hidden Layers")
#plt.ylabel("Accuracy (%)")
#plt.title("Nursery - Neural Network")


plt2.plot(car_x,car_y, label="Training")
plt2.plot(testcx, testcy, label = "Cross Validation")
plt2.xlabel("Number of Hidden Layers")
plt2.ylabel("Accuracy (%)")

plt2.legend(loc='upper left', frameon=False)
plt2.title("Car - Neural Network")