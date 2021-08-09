q# -*- coding: utf-8 -*-
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

def train(X,y, scores=False):

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=1)
    tree = DecisionTreeClassifier()
    naive = GaussianNB()


    # fit the model
    knn.fit(X_train, y_train)
    preds_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, preds_knn)
    #roc_knn = roc_auc_score(y_test, preds_knn, multi_class='ovr')
    print(f'Successfully trained model with an accuracy of {acc_knn:.2f}')
    #print(f'Successfully trained model with an roc_auc score of {roc_knn:.2f}')

    # fit the model
    tree.fit(X_train, y_train)
    preds_tree = tree.predict(X_test)
    acc_tree = accuracy_score(y_test, preds_tree)
    #roc_tree = roc_auc_score(y_test, preds_tree, multi_class='ovr')
    print(f'Successfully trained model with an accuracy of {acc_tree:.2f}')
    #print(f'Successfully trained model with an roc_auc score of {roc_tree:.2f}')

    # fit the model
    naive.fit(X_train, y_train)
    preds_tree = naive.predict(X_test)
    acc_naive = accuracy_score(y_test, preds_tree)
    #roc_tree = roc_auc_score(y_test, preds_tree, multi_class='ovr')
    print(f'Successfully trained model with an accuracy of {acc_naive:.2f}')
    #print(f'Successfully trained model with an roc_auc score of {roc_tree:.2f}')

    return knn, tree, naive, acc_knn, acc_tree, acc_naive


if __name__ == '__main__':

    iris_data = datasets.load_iris()
    X = iris_data['data']
    y = iris_data['target']

    labels = {0 : 'iris-setosa',
              1 : 'iris-versicolor',
              2 : 'iris-virginica'}

    # rename integer labels to actual flower names
    y = np.vectorize(labels.__getitem__)(y)

    mdl_knn, mdl_tree, naive, acc_knn, acc_tree, acc_naive = train(X,y, scores=True)

    data = {}
    data['models'] = []
    data['models'].append({
        'name' : 'KNN',
        'accuracy': acc_knn
        })
    data['models'].append({
        'name' : 'Tree',
        'accuracy': acc_tree
        })
    data['models'].append({
        'name' : 'NaiveBayes',
        'accuracy': acc_naive
        })
    
    with open('accuracy.txt', 'w') as the_file:
        json.dump(data, the_file)
   