from __future__ import print_function
__docformat__ = 'restructedtext en'
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):
    """
    Logistic Regression using Stochastic Gradient Descent
    Source: http://deeplearning.net/tutorial/logreg.html
    """
    def __init__(self, input, n_in, n_out):
        # Initialize W (n_in by n_out) with 0s
        self.W = theano.shared(
            value = np.zeros((n_in, n_out), dtype = theano.config.floatX),
            name = 'W',
            borrow = True
        )

        # Initialize b with 0s
        self.b = theano.shared(
            value = np.zeros((n_out,), dtype = theano.config.floatX),
            name = 'b',
            borrow = True
        )

        # Computing class membership probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # Computing prediction as class with maximal probability
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        # Model parameters
        self.params = [self.W, self.b]

        # Tracking model input
        self.input = input

    def negative_log_likelihood(self, y):
        """
        Returns the mean of the negative log-likelihood of the prediction of
        this model under a given target distribution.
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Returns a float representing the number of errors in the minibatch over
        the total number of examples of the minibatch.
        """
        # Check dimensions of y and y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # Check y datatype
        if y.dtype.startswith('int'):
            # T.neq returns a vector of 0s and 1s, where 1 = error
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

"""
Testing
"""
# Symbolic variables for input (x and y represent a minibatch)
x = T.matrix('x') # Images
y = T.ivector('y')  # Labels

# Construct Logistic Regression class
# TODO: Change from MNIST to 60x60 for Kaggle data
classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

# Cost to minimize is the neg. log likelihood of the model in symbolic format
cost = classifier.negative_log_likelihood(y)

# Gradients (symbolic variables)
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

# Specify how to update the parameters of the model as a list of
# (variable, update expression) pairs
updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]

# Compiling a Theano function `train_model` that returns the cost, but in
# the same time updates the parameter of the model based on the rules
# defined in `updates`
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# Compiling a Theano function that computes the mistakes that are made by
# the model on a minibatch
test_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
