import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data.preprocess import get_data, get_data_keras

# Get the data
n = 100000
n_perturbed = 0

X, y = get_data_keras(n, n_perturbed)
# Reshape data as (example, features)
X = X.reshape((n+n_perturbed,60*60))

# Train classifier
print "Training classifier..."
clf = SGDClassifier(loss = 'log');
clf.fit(X, y)
print "Done."

X_valid, y_valid = X[:10000], y[:10000]
#X_valid, y_valid = get_data_keras(1000)

predictions = clf.predict(X_valid)

print predictions[:100]
print accuracy_score(y_valid, predictions)
