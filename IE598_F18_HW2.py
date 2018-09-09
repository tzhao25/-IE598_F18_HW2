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
from matplotlib.colors import ListedColormap
import numpy as np
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
    knn = KNeighborsClassifier(n_neighbors = k, p=2, metric='minkowski')
    
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
plt.show()    
    
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
    plt.scatter(md, accuracy, c='red')
    plt.title('Max Depth vs Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
plt.show()
    

## Implement the Code from Chapter 3    
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
iris = datasets.load_iris() 
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler() 
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o', s=55, label='test set')
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()











