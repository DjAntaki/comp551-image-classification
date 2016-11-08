import numpy as np
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import geometric_transform
import utils

def get_data_train_valid(n=100000, n_perturbed = 0, train=0.75, shuffle=False, threshold=False,show = False):
    """
    Returns n training examples and their corresponding labels splitted in train and validation set.
    n_perturbed images and labels are appended to the training set if specified.

    """
    # Images (60x60)
#    train_X = np.fromfile('../data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = np.fromfile('data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = train_X.reshape((n,60,60))

    if not threshold is False :
        assert type(threshold) is int
        train_X = contrast_threshold(train_X,threshold)

    # Labels
#    train_y = pd.read_csv('../data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    train_y = pd.read_csv('data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]

    # Splitting data in train and validation set
    train_set, valid_set = utils.split_train_valid(zip(train_X, train_y), train=train, shuffle=shuffle)
    X_train, y_train = map(np.array, zip(*train_set))
    X_valid, y_valid = map(np.array, zip(*valid_set))

    # Perturb data if specified
    X_train, y_train = perturb_data(X_train, y_train, n, n_perturbed)

    if (show and n_perturbed > 0):
        print "Displaying example perturbation (skewing)"
        plt.figure()
        plt.imshow(np.concatenate((X_train[0], np.zeros((60,60)), X_train[n]), axis=-1), cmap="gray")
        plt.show()

    if (show and not threshold is False):
        print "Displaying example filtering with threshold"
        plt.figure()
        plt.imshow(np.concatenate((X_train[0], X_train[1], X_train[2]), axis=-1), cmap="gray")
        plt.show()

    return X_train, y_train, X_valid, y_valid

def get_test_data(show=False,threshold=False):
    """
    """
    # Images (60x60)
    train_X = np.fromfile('../data/test_x.bin', dtype='uint8')
#    train_X = np.fromfile('data/test_x.bin', dtype='uint8')

    if not threshold is False :
        assert type(threshold) is int
        train_X = contrast_threshold(train_X,threshold)

    n = len(train_X) / 3600
    X_train = train_X.reshape((n, 60, 60))

    if show:
        plt.figure()
        plt.imshow(np.concatenate((X_train[0], np.zeros((60, 60)), X_train[-1]), axis=-1), cmap="gray")
        plt.show()

    return X_train

def get_data(n=100000, n_perturbed = 0, show = False):
    """
    Returns n training examples and their corresponding labels.
    n_perturbed images and labels are appended if specified.
    """
    # Images (60x60)
#    train_X = np.fromfile('../data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = np.fromfile('data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = train_X.reshape((n,60,60))
    # Labels
#    train_y = pd.read_csv('../data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    train_y = pd.read_csv('data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]

    # Perturb data if specified
    X, y = perturb_data(train_X, train_y, n, n_perturbed)

    if (show and n_perturbed > 0):
        print "Displaying example perturbation (skewing)"
        plt.figure()
        plt.imshow(np.concatenate((X[0], np.zeros((60,60)), X[n]), axis=-1), cmap="gray")
        plt.show()

    return X, y

def contrast_threshold(X, threshold=254):
    """Sets all pixels under the given threshold to 0."""
    X[X < threshold] = 0
    return X


def perturb_data(X, y, n, n_perturbed):
    """
    Simple data perturbation using a normalized skewing procedure.
    """
    if (n_perturbed > 0):
        X_all = X.copy() # All original examples
        print "Perturbing data..."
        X_perturbed = X[0:n_perturbed].copy() # Number of examples to copy
        X_perturbed = np.concatenate([X_all, perturb_skew(X_perturbed)])
        y_perturbed = np.concatenate([y, y[0:n_perturbed]])
        print "Perturbation complete."
        return X_perturbed, y_perturbed
    else:
        return X, y

def perturb_skew(X):
    """
    Skew the provided images.
    """
    X_skewed = X.copy()
    for i, image in enumerate(X_skewed):
        X_skewed[i] = skew(image)
    return X_skewed

def skew(image):
    """
    Skew the provided image.
    Source: http://stackoverflow.com/a/33088550/4855984
    """
    h, l = image.shape
    dl = np.random.normal(loc=15, scale=5) # Norm = 15px, Sigma = 5px

    def mapping(lc):
        l, c = lc
        dec = (dl*(l-h))/h
        return l, c+dec
    return geometric_transform(image, mapping, (h, l), order=5, mode='nearest')

#get_data(10,10,True)
#get_test_data()

get_data_train_valid(n=100,show=True,threshold=254)