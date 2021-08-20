"""
Classes implementing various regularisation functions
"""

import numpy as np


class Regulariser:
    def __init__(self):
        pass
    def get_penalty_term(self):
        pass
    def get_W_term(self):
        pass
    def get_b_term(self):
        pass

class L1_Regulariser(Regulariser):
    """
    Lasso regularisation; favours sparse weights
    """
    def __init__(self, lamb):
        """

        :param lamb: lambda value
        """
        self.lamb = lamb

    def get_penalty_term(self, params):
        return self.lamb * np.sum(np.abs(params))

class L2_Regulariser(Regulariser):
    """
    Ridge Regularisation; favours minimised weights without extreme values (evenly distributed weights)
    """
    def __init__(self, lamb):
        self.lamb = lamb

    def get_penalty_term(self, params):
        return self.lamb * np.sum(params ** 2)

