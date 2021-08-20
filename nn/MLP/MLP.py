"""
Construct and train a simple MLP model
"""

from global_params import *
import numpy as np
import pickle
from matplotlib import pyplot as plt
from nn.MLP.activation import Sigmoid, Softmax, ReLU
from nn.MLP.loss import CrossEntropy
from nn.MLP.regularise import L1_Regulariser
from nn.MLP.layer import Full, Output
from nn.MLP.train import Trainer


class MLP:
    """
    Currently, a pre-defined MLP with one hidden layer and an output layer. Demonstrates how to use the layer objects
    to build an MLP of arbitrary dimensions.
    """
    def __init__(self, n_inputs, layer_size, n_classes, lamb=None):
        """
        Initialise.
        :param n_inputs: int, the size of the input vector
        :param layer_size: int, the size of the hidden layer
        :param n_classes: int, the number of output classes
        :param lamb: optional float, the lambda parameter for regularisation
        """
        self.full1 = Full(n_inputs, layer_size, activation=ReLU)
        self.out1 = Output(layer_size, n_classes)

        self.regularise = False
        if lamb is not None:
            self.lamb = lamb
            self.regularise = True

    def forward(self, X):
        """
        Pass a batch of data samples through the network
        :param X: numpy array of shape (n_inputs, N), where N is the number of samples
        :return: numpy array of shape (n_classes, N)
        """
        return self.out1.forward(self.full1.forward(X))

    def backward(self, Y, eta=0.1):
        """
        Pass true labels backward through network, compute associated errors, and update weights accordingly
        :param Y: numpy array of shape (n_classes, N) where N is the number of samples
        :param eta: optional float, the learning rate
        :return:
        """
        self.full1.backward(self.out1.backward(Y))
        self.out1.update_params(eta)
        self.full1.update_params(eta)

    def fit_step(self, X, Y, eta=0.1):
        """
        One step of passing a set of samples through and updating the parameters based on the errors
        :param X: numpy array of shape (n_inputs, N) where N is the number of samples
        :param Y: numpy array of shape (n_classes, N)
        :param eta: optional float, the learning rate
        :return:
        """
        self.forward(X)
        self.backward(Y, eta)


def load_data(filename):
    """
    Load CIFAR-10 dataset
    :param filename: string, path of CIFAR file to load
    :return: numpy arrays X, Y, y
    """
    with open(filename, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    X = dict[b'data'].astype('float64') / 255.
    y = np.asarray(dict[b'labels'])
    Y = np.zeros((X.shape[0], 10))
    Y[np.arange(y.size), y] = 1  # created one-hot encodings using indices y label array

    return X, Y, y

if __name__ == "__main__":
    trainX, trainY, trainy = load_data("../../data/cifar10/data_batch_1")
    valX, valY, valy = load_data("../../data/cifar10/data_batch_2")

    # normalise the data to zero-mean
    mean_X = np.mean(trainX, axis=0)
    std_X = np.std(trainX, axis=0)

    trainX -= mean_X
    trainX /= std_X
    valX -= mean_X
    valX /= std_X

    d = trainX.shape[1]
    K = 10

    trainX, trainY, valX, valY = trainX.transpose(), trainY.transpose(), valX.transpose(), valY.transpose()  # madel
    # takes data matrix in format d x N, where d is the dimension of the data and N is the number of samples

    # initialise model
    eta = 0.1
    batch_size = 32
    epochs = 20
    model = MLP(d, 20, K)
    loss = CrossEntropy()
    trainer = Trainer(model, loss, eta=eta, batch_size=batch_size, epochs=epochs)

    # train model
    history = trainer.train(trainX, trainY, valX, valY)

    # plot results
    plt.plot(history['train_loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()

