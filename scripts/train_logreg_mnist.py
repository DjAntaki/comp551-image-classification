from data.preprocess import get_data, get_data_keras
import numpy as np
from skimage.transform import resize
from operator import itemgetter
from models import log_reg
from sklearn.metrics import accuracy_score

# Validation Examples
n_valid = 1000 # Max 100000
# Model Configuration
learning_rate = 0.09
n_epochs = 1000
batch_size = 1

dataset = get_data(n_valid)
X_valid, y_valid = dataset
X_valid = X_valid*1.0

print("Building logistic regression model using MNIST dataset...")
log_reg.sgd_optimization_MNIST();
print("Done.")

print("Downsizing validation set...")
num_example = len(X_valid)
X_d = np.zeros(shape=(num_example,1,28,28),dtype="float32")
for i, x in enumerate(X_valid):
    X_d[i] = resize(x,(28,28))
print("Done.")

print("Predicting...")
predictions = log_reg.predict_MNIST(X_d)
print("Accuracy score:")
print accuracy_score(y_valid, predictions)
