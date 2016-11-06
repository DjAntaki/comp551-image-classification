from data.preprocess import get_data, get_data_keras
from models import log_reg
import numpy as np
import six.moves.cPickle as pickle
from skimage.transform import resize
from sklearn.metrics import accuracy_score

# Training Configuration
n_train = 100000 # Max 100000
n_perturbed = 100000
# Model Configuration
learning_rate = 0.09
n_epochs = 1000
batch_size = 1
load = False

if (load):
    print "Loading data..."
    pickle_file = open('./data/keras_data.pkl', 'rb')
    dataset = pickle.load(pickle_file)
    pickle_file.close()
    X, y = dataset
    print "Done."
else:
    dataset = get_data_keras(n_train, n_perturbed)
    X, y = dataset

print("Building logistic regression model")
log_reg.sgd_optimization(dataset, learning_rate, n_epochs, batch_size)

X_valid, y_valid = X[:1000], y[:1000]
#X_valid, y_valid = get_data_keras(1000)

predictions = log_reg.predict(X_valid)
print accuracy_score(y_valid, predictions)
