import numpy as np
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import geometric_transform

def get_data(n=10000):
    """
    Returns 2n training examples and their corresponding labels.
    The second half (n+1 to 2n) consists of perturbed images using skewing.
    """
    # Images (60x60)
    train_X = np.fromfile('train_x.bin', count=(n*60*60), dtype='uint8')
    train_X = train_X.reshape((n,60,60))
    # Labels
    train_y = pd.read_csv('train_y.csv', delimiter=',', index_col=0, engine='python').values[:n]
    # Perturb data
    perturbed_X, perturbed_y = perturb_data(train_X, train_y, n)

    print "Displaying example perturbation (skewing)"
    plt.figure()
    plt.imshow(np.concatenate((train_X[0],np.zeros((60,60)),perturbed_X[n]),axis=-1),cmap="gray")
    plt.show()

    return perturbed_X, perturbed_y

def perturb_data(X, y, n):
    """
    Simple data perturbation using a normalized skewing procedure.
    """
    print "Perturbing data..."
    X_perturbed = X.copy()
    X_perturbed = np.concatenate([X_perturbed, perturb_skew(X_perturbed)])
    y_perturbed = np.concatenate([y]*2)
    print "Perturbation complete."

    return X_perturbed, y_perturbed

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
