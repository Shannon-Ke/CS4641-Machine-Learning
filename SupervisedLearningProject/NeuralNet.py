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

#for i in range(1,100):
#    ## Splitting up features and target for nursery
#     # and making training and test data sets
#    X = nursery.values[:, 0:7]
#    Y = nursery.values[:, 8]
#    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = (100 - i)/100.0,
#                                                        random_state = 100)
#    ## Splitting up features and target for car
#     # and making training and test data sets
#    W = car.values[:, 0:5]
#    Z = car.values[:, 6]
#    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = (100 - i)/100.0,
#                                                        random_state = 100)
#    
#    scaler = StandardScaler()
#    scaler.fit(X_train)
#    X_train = scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#    
#    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
#    mlp.fit(X_train,y_train)
#    
#    predictions = mlp.predict(X_test)
#    #print(confusion_matrix(y_test,predictions))
#    #print(classification_report(y_test,predictions))
#    #print "Accuracy for nurse: ", accuracy_score(y_test,predictions)*100
#    ##
#    nursery_x.append(i/100.0)
#    nursery_y.append(accuracy_score(y_test,predictions)*100)
#    
#    
#    
#    scaler.fit(W_train)
#    W_train = scaler.transform(W_train)
#    W_test = scaler.transform(W_test)
#    
#    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
#    mlp.fit(W_train,z_train)
#    
#    predictions = mlp.predict(W_test)
#    car_x.append(i/100.0)
#    car_y.append(accuracy_score(z_test,predictions)*100)
#    #print(confusion_matrix(z_test,predictions))
#    #print(classification_report(z_test,predictions))
    

avg = 0.0
numcalled = 0

y = []
for i in range (1,14):
    W = car.values[:, 0:5]
    Z = car.values[:, 6]
    W_train, W_test, z_train, z_test = train_test_split( W, Z, test_size = 0.25,
                                                            random_state = 100)
    numcalled += 1
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    pca = PCA(0.7 + 0.02 * i) #.99 is best value
    pca.fit(W_train)
    
    W_train = pca.transform(W_train)
    W_test = pca.transform(W_test)
    mlp.fit(W_train,z_train)
    
    prediction = mlp.predict(W_train)
    print(accuracy_score(z_train, prediction)*100)
    predictions = mlp.predict(W_test)
    car_x.append(0.7 + i * 0.02)
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
plt.xlabel('PCA variance %')
plt.ylabel('Accuracy')
plt.legend(loc='upper left', frameon=False)
plt.title('PCA with the Cars Dataset on a NN')