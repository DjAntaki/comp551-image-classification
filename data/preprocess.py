import numpy as np
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import json
from scipy.ndimage.interpolation import geometric_transform
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def get_data(n=100000, n_perturbed = 0, show = False):
    """
    Returns n training examples and their corresponding labels.
    n_perturbed images and labels are appended if specified.
    """
    print "Getting data..."
    # Images (60x60)
    train_X = np.fromfile('./data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = train_X.reshape((n,60,60))
    # Labels
    train_y = pd.read_csv('./data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    # Perturb data if specified
    X, y = perturb_data(train_X, train_y, n, n_perturbed)
    print "Done."

    if (show and n_perturbed > 0):
        print "Displaying example perturbation (skewing)"
        plt.figure()
        plt.imshow(np.concatenate((X[0], np.zeros((60,60)), X[n]), axis=-1), cmap="gray")
        plt.show()

    return X, y

def get_data_keras(n=100000, n_perturbed = 0, save_data = False):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        zca_whitening=True)

    print "Getting data..."
    # Images (60x60)
    train_X = np.fromfile('./data/train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = train_X.reshape(n,1,60,60).astype('float32')
    # Labels
    train_y = pd.read_csv('./data/train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    train_y = train_y.reshape(n,).astype('float32')
    print "Done."
    print "Fitting data..."
    datagen.fit(train_X)
    for X_batch, y_batch in datagen.flow(train_X, train_y, batch_size=n):
        #print "Displaying example perturbation (Keras)"
        #print y_batch[0]
        #plt.figure()
        #plt.imshow(X_batch[0].reshape(60,60), cmap="gray")
        #plt.show()
        X_batch = X_batch.reshape(n,60,60)

        print "Done."
        if (save_data and n_perturbed == 0):
            print "Saving dataset..."
            with open('./data/keras_data.pkl', 'wb') as f:
                pickle.dump((X_batch, y_batch), f)
                print "Done."

        if (n_perturbed == 0):
            return X_batch, y_batch
        else:
            return add_keras_perturbed(X_batch, y_batch, n_perturbed, save_data)

def add_keras_perturbed(X, y, n, save_data):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    print "Perturbing data..."
    for X_batch, y_batch in datagen.flow(X, y, batch_size=n):
        #print "Displaying example perturbation (Keras)"
        #print y_batch[0]
        #plt.figure()
        #plt.imshow(X_batch[0].reshape(60,60), cmap="gray")
        #plt.show()
        X_batch = X_batch.reshape(n,60,60)
        X = np.concatenate([X, X_batch])
        y = np.concatenate([y, y_batch])
        print "Done."

        if (save_data):
            print "Saving dataset..."
            with open('./data/keras_data_perturbed.pkl', 'wb') as f:
                pickle.dump((X, y), f)
                print "Done."

        return X, y

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
