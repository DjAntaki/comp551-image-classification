from data.preprocess import get_data
from models import log_reg
import numpy as np

# Training Configuration
n_train = 100000 # Max 100000
# Model Configuration
learning_rate = 0.13
n_epochs = 1000
dataset = get_data(n_train) # (X, y)
batch_size = 1000

print("Building logistic regression model")
log_reg.sgd_optimization(dataset, learning_rate, n_epochs, batch_size)
