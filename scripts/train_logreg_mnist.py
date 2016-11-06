from data.preprocess import get_data, get_data_keras
import numpy as np
from skimage.transform import resize
from operator import itemgetter
from models import log_reg
from sklearn.metrics import accuracy_score

# Training Configuration
n_train = 100000 # Max 100000
# Model Configuration
learning_rate = 0.09
n_epochs = 1000
batch_size = 1
load = True

if (load):
    print "Loading data..."
    pickle_file = open('./data/keras_data.pkl', 'rb')
    dataset = pickle.load(pickle_file)
    pickle_file.close()
    X, y = dataset
    print "Done."
else:
    dataset = get_data_keras(n_train)
    X, y = dataset

print("Building logistic regression model")
#log_reg.sgd_optimization_MNIST();

X_valid, y_valid = X[:1000], y[:1000]

#Downsizing...
num_example = len(X_valid)
X_d = np.zeros(shape=(num_example,1,28,28),dtype="float32")
for i, x in enumerate(X_valid):
    X_d[i] = resize(x,(28,28))

predictions = log_reg.predict_MNIST(X_d)
print accuracy_score(y_valid, predictions)
