"""
Classes implementing various activation functions
"""

import numpy as np


class Activation:
    def __init__(self):
        pass

    def compute(self, x):
        """
        Default identity function
        :param x: numpy array of a vector
        :return:
        """
        return x

    def compute_der(self, x):
        return np.ones(x.shape)

class Sigmoid(Activation):
    """
    Implements the Sigmoid activation function and derivative version thereof.
    """
    def compute(self, x):
        return 2 / (1 + np.exp(-x)) - 1
    def compute_der(self, x):
        return ((1 + x) * (1 - x)) / 2

class ReLU(Activation):
    """
    Implements the ReLU activation function and the derivative thereof.
    """
    def compute(self, x):
        x[x < 0] = 0
        return x
    def compute_der(self, x):
        x[x>0] = 1
        x[x<=0] = 0
        return 0

class Softmax(Activation):
    """
    Implements the Softmax activation function (recommended for output layers only).
    """
    def compute(self, x):
        denom = np.sum(np.exp(x), axis=0)
        return np.exp(x) / denom
    def compute_der(self, x):
        pass

