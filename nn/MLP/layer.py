"""
Classes implementing various layer types
"""

import numpy as np
from nn.MLP.activation import Sigmoid, Softmax, ReLU


class Layer:
    """
    Generic layer class for a single layer in a network model.
    """
    def __init__(self, n_inputs, size, activation=Sigmoid):
        """

        :param n_inputs: int, the size of the input vector
        :param size: int, the number of units in this layer ( = size of the output vector)
        :param activation: Activation class object, the activation function to use
        """
        self.W = np.random.normal(0, 1/np.sqrt(n_inputs), (size, n_inputs))
        self.b = np.ones((size, 1))

        self.X = np.zeros((n_inputs, 1))
        self.A = np.zeros((size, 1))  # activations
        self.deltaW = np.zeros((size, n_inputs))
        self.deltab = np.zeros((size, 1))

        self.activation = activation()

    def forward(self, X):
        """
        Standard layer operation: multiply input vector with weight matrix, add bias, and compute activation.
        :param X: numpy array of shape (n_inputs, N), where N is the number of samples
        :return: numpy array of shape (size, N)
        """
        self.X = X
        S = np.matmul(self.W, X) + self.b
        self.A = self.activation.compute(S)
        return self.A

    def backward(self, Y):
        return Y

    def set_gradients(self, W_term=0, b_term=0):
        """
        Update the gradients. When using regularising terms, assign these to W_term and b_term accordingly.
        :param W_term:
        :param b_term:
        :return:
        """
        self.W_grad = self.deltaW + W_term
        self.b_grad = self.deltab + b_term

    def update_params(self, eta=0.1, W_term=0, b_term=0):
        """
        Update the W and b parameters based on the computed gradients
        :param eta: float (typically in range 0 - 1); the learning rate used
        :param W_term: optional float, a term added to the W gradient calculation. Used to add regularisation terms.
        :param b_term: optional float, a term added to the b gradient calculation. Used to add regularisation terms.
        :return:
        """
        self.set_gradients(W_term, b_term)
        self.W -= eta * self.W_grad
        self.b -= eta * self.b_grad

class Full(Layer):
    def backward(self, Y):
        """
        Pass errors backward and compute error gradients along the way
        :param Y: numpy array of shape (size, N), typically the errors of the layer this layer outputs to
        :return: numpy array of shape (n_inputs, N); the errors of this layer
        """
        n = self.X.shape[1]  # number of samples in batch
        self.A = self.activation.compute_der(self.A)
        G = Y * self.A
        self.deltaW = 1/n * np.matmul(G, self.X.transpose())
        self.deltab = 1/n * np.matmul(G, np.ones((n, 1)))
        G = np.matmul(self.W.transpose(), G)
        return G

class Output(Full):
    """
    The output layer for a neural network, which needs to treat some operations differently.
    """
    def __init__(self, n_inputs, size, activation=Softmax):
        """
        Initialisation
        :param n_inputs: int, the size of the input vector
        :param size: int, the number of units in this layer ( = size of the output vector)
        :param activation: Activation class object, the activation function to use
        """
        self.W = np.random.normal(0, 1 / np.sqrt(n_inputs), (size, n_inputs))
        self.b = np.ones((size, 1))

        self.X = np.zeros((n_inputs, 1))
        self.A = np.zeros((size, 1))  # activations
        self.deltaW = np.zeros((size, n_inputs))
        self.deltab = np.zeros((size, 1))

        self.activation = activation()

    def backward(self, Y):
        n = self.X.shape[1]  # number of samples in the batch
        G = -(Y - self.A)
        self.deltaW = 1/n * np.matmul(G, self.X.transpose())
        self.deltab = 1/n * np.matmul(G, np.ones((n, 1)))
        G = np.matmul(self.W.transpose(), G)
        return G
