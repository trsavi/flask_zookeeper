# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:16:31 2021

@author: Pc4y
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:20:55 2021

@author: Pc4y
"""

# model.py
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import numpy as np
import json

def train():
    
    iris_data = datasets.load_iris()
    X = iris_data['data']
    y = iris_data['target']

    labels = {0 : 'iris-setosa',
              1 : 'iris-versicolor',
              2 : 'iris-virginica'}

    # rename integer labels to actual flower names
    y = np.vectorize(labels.__getitem__)(y)


    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=1)
    tree = DecisionTreeClassifier()
    naive = GaussianNB()

    # fit the model
    knn.fit(X_train, y_train)
    
    # fit the model
    tree.fit(X_train, y_train)
    

    # fit the model
    naive.fit(X_train, y_train)

    # serialize model
    pickle.dump(knn, open('iris_knn.pkl', 'wb'))
    # serialize model
    pickle.dump(tree, open('iris_tree.pkl', 'wb'))
    # serialize model
    pickle.dump(naive, open('iris_naive.pkl', 'wb'))
    print('Successful!')


if __name__ == '__main__':
    train()
    