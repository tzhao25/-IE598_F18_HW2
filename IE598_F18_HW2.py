# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:10:09 2018

@author: zhth1202
""" 

print("My name is Tianhao Zhao")
print("My NetID is: tzhao25")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print('')
import matplotlib.pyplot as plt
import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))

from sklearn import datasets

iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# K Neighbors Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
# Get dataset with only the first two attributes
X, y = X_iris, y_iris
# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

k_range = range(1,26)
score = []
score_train = []
score_test = []
print ('The accuracy of K Nearest Neighbors model is: ')
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    
    knn.fit(X_train, y_train)
    knn.fit(X_test, y_test)
    
    score_train = knn.score(X_train, y_train)
    score_test = knn.score(X_test, y_test)
    print ('Train set: k =', k, 'Score:', score_train)
    print ('Test set: k =', k, 'Score:', score_test)
    plt.scatter(k, score_train, c='red')
    plt.scatter(k, score_test, c='blue')
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_iris, y_iris)
    score = knn.score(X_iris, y_iris)
    print ('No Training: k =', k, 'Score:', score)
    plt.scatter(k, score, c='green')
    plt.title('K vs Scores')
    plt.legend(["Train set", "Test set", "No training set"])
    plt.xlabel('k value')
    plt.ylabel('scores')
    
    
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

X, y = X_iris, y_iris
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
md_range = range(2,6)
for md in md_range:
    dt = DecisionTreeClassifier(max_depth = md, criterion = 'entropy', random_state = 1)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('The accuracy of Decision tree model with max depth of', md, 'is ', accuracy)











