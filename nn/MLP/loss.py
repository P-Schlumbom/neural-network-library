"""
Classes implementing various loss functions
"""

import numpy as np


class Loss:
    def __init__(self):
        pass
    def compute(self, P, Y):
        return np.mean(P-Y)

class CrossEntropy(Loss):
    def compute(self, P, Y):
        """
        compute the cross entropy loss of the predicted labels
        :param P: numpy array of shape Kxn, n samples of label predictions over K classes
        :param Y: numpy array of shape Kxn, one-hot true labels
        :return: float, crossentropy loss of predictions
        """
        probs =  np.sum(P * Y, axis=0)
        l_crosses = -np.log(probs)
        return l_crosses

