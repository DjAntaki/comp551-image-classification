import numpy as np
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
from scipy import ndimage
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
from keras.preprocessing.image import ImageDataGenerator
from skimage import feature, filters
from skimage.morphology import disk

def get_data(n=100000, n_perturbed = 0, show = False):
    """
    Returns n training examples and their corresponding labels.
    n_perturbed images and labels are appended if specified.
    """
    print "Getting data..."
    # Images (60x60)
#    train_X = np.fromfile('../data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = np.fromfile('data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = train_X.reshape((n,60,60))
    # Labels
#    train_y = pd.read_csv('../data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    train_y = pd.read_csv('data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]

    # Perturb data if specified
    X, y = perturb_data(train_X, train_y, n, n_perturbed)
    print "Done."

    if (show):
        print "Displaying example perturbation (skewing)"
        plt.figure()
        for i in range(0, 9):
            #edges = feature.canny(X[i], sigma = 1)
            #threshold_global_otsu = filters.threshold_otsu(X[i])
            #res = X[i] >= threshold_global_otsu
            #res = X[i] >= 255
            #sx = ndimage.sobel(im, axis=1)
            #sy = ndimage.sobel(im, axis=0)
            #sob = np.hypot(sx, sy)
            #sob *= 255.0 / np.max(sob)
            plt.subplot(330 + 1 + i)
            plt.imshow(X[i], cmap="gray")
        plt.show()
    if (show and n_perturbed > 0):
        plt.figure()
        plt.imshow(np.concatenate((X[0], np.zeros((60,60)), X[n]), axis=-1), cmap="gray")
        plt.show()

    return (X[:] >= 255)*1, y

def get_data_keras(n=100000, n_perturbed = 0, show = False):
    """
    Returns n training examples and their corresponding labels.
    Applies ZCA whitening (including feature normalization and scaling).
    n_perturbed images and labels are appended if specified.
    """
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False)

    print "Getting data..."
    # Images (60x60)
    train_X = np.fromfile('./data/train_x.bin', count=(n*60*60), dtype='uint8')
    # Filter
    train_X[:] = (train_X[:] >= 255)*1
    train_X = train_X.reshape(n,1,60,60).astype('float32')
    # Labels
    train_y = pd.read_csv('./data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    train_y = train_y.reshape(n,).astype('float32')
    print "Done."

    print "Fitting data..."
    datagen.fit(train_X)
    for X_batch, y_batch in datagen.flow(train_X, train_y, batch_size=n):
        if (show):
            print "Displaying example perturbation (Keras)"
            plt.figure()
            for i in range(0, 9):
                plt.subplot(330 + 1 + i)
                plt.imshow(X_batch[i].reshape(60,60), cmap="gray")
            plt.show()
        X_batch = X_batch.reshape(n,60,60)
        print "Done."

        if (n_perturbed == 0):
            return X_batch, y_batch
        else:
            return add_keras_perturbed(X_batch, y_batch, n_perturbed, show)

def add_keras_perturbed(X, y, n, show):
    """
    Apply image perturbation using the specified Keras datagen parameters.
    """
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

    print "Perturbing data..."
    X_2 = X.reshape(X.shape[0],1,60,60).astype('float32')
    for X_batch, y_batch in datagen.flow(X_2, y, batch_size=n):
        if (show):
            print "Displaying example perturbation (Keras)"
            plt.figure()
            for i in range(0, 9):
                plt.subplot(330 + 1 + i)
                plt.imshow(X_batch[i].reshape(60,60), cmap="gray")
            plt.show()
        X_batch = X_batch.reshape(n,60,60)
        X = np.concatenate([X, X_batch])
        y = np.concatenate([y, y_batch])
        print "Done."

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
    total = len(X)
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

#get_data_train_valid(n=100,show=True,threshold=254)
