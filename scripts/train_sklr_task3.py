import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from data.preprocess import get_data, get_data_keras

# Get the data
n = 100000
n_perturbed = 0

#X, y = get_data_keras(n, n_perturbed)
X, y = get_data_keras(n, n_perturbed)
# Reshape data as (example, features)
X = X.reshape((n+n_perturbed,60*60))
y = y.reshape((n+n_perturbed,))

def crossValidate(X, y):
    print "Cross-validating..."
    kf = KFold(n_splits = 10)
    count = 1
    average = []

    clf = SGDClassifier(loss = 'log')
    for train, valid in kf.split(X):
        print "k-fold #" + str(count)
        clf.fit(X[train], y[train])
        print "Predicting..."
        predictions = clf.predict(X[valid])
        accuracy = accuracy_score(predictions, y[valid])
        print "Accuracy score: ",
        print accuracy
        average.append(accuracy)
        count += 1

    average = sum(average) / len(average)
    print "Average accuracy score: ",
    print average

crossValidate(X, y)

#clf = SGDClassifier(loss = 'log')
#clf.fit(X[1000:], y[1000:])
#predictions = clf.predict(X[:1000])
#accuracy = accuracy_score(predictions, y[:1000])
#print "Accuracy score: ",
#print accuracy
