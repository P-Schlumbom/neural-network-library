import numpy as np
import pickle
from matplotlib import pyplot as plt

VERBOSE = 1

#----------------------------ACTIVATION CLASSES-------------------------------#
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
#-----------------------------------------------------------------------------#

#----------------------------LOSS CLASSES-------------------------------------#
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
#-----------------------------------------------------------------------------#

#------------------------------REGULARISATION CLASSES-------------------------#
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

#-----------------------------------------------------------------------------#

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

class Trainer:
    """
    Given a model and training parameters, handle the training of the model on a given set of data X, Y
    """
    def __init__(self, model, loss, eta=0.1, batch_size=10, epochs=1, steps=None):
        """
        Initialisation.
        :param model: MLP object, the model to train
        :param loss: Loss object, the loss function to use
        :param eta: optional float, the learning rate
        :param batch_size: optional int, the size of the batches to load from the data
        :param epochs: optional int, the number of epochs to train for
        :param steps: optional int, the number of steps to train for (note this will only be used if epochs is None)
        """
        self.model = model
        self.loss = loss
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps

        self.iterate_epoch = True if self.epochs is not None else False

    def train(self, X_train, Y_train, X_val=None, Y_val=None):
        """
        Train the provided model on the given training data, and optionally validate model performance during training
        with validation data sets.
        :param X: numpy array of shape (d, N) where d is the dimensionality of the input points and N is the number of
        samples in the dataset
        :param Y: numpy array of shape (K, N) where K is the number of prediction classes and N is the number of samples
         in the dataset
        :return: dict, a dictionary of lists recording the training and validation losses (and accuracies) per epoch
        """
        validate = X_val is not None  # only validate if validation set is provided
        N = X_train.shape[1]
        if self.iterate_epoch:
            self.steps = int(self.epochs * (N // self.batch_size))
        else:
            self.epochs = int((self.steps // (N / self.batch_size)) + 1)

        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        step = 0
        for i in range(self.epochs):
            training_loss = 0
            for j in range(0, N, self.batch_size):
                # select batch
                j_start = j
                j_end = min(j + self.batch_size, N)
                Xbatch = X_train[:, j_start:j_end]
                Ybatch = Y_train[:, j_start:j_end]

                self.model.fit_step(Xbatch, Ybatch, self.eta)
                training_loss += np.sum(self.loss.compute(self.model.forward(Xbatch), Ybatch))

                step += 1
                if not self.iterate_epoch and step > self.steps:  # if training time is defined by number of steps
                    # instead of epochs, stop training once number of steps is exceeded,
                    break

            # compute end of epoch info
            training_loss /= N  # average loss over whole training set
            validation_loss = None
            if validate:
                validation_loss = np.sum(self.loss.compute(self.model.forward(X_val), Y_val))
                validation_loss /= X_val.shape[1]
            history['epoch'].append(i)
            history['train_loss'].append(training_loss)
            history['val_loss'].append(validation_loss)
            if VERBOSE >= 1:
                print("epoch {}: train_loss = {:.5g}, val_loss = {:.5g}".format(i, training_loss, validation_loss))

        return history

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
    trainX, trainY, trainy = load_data("../data/cifar10/data_batch_1")
    valX, valY, valy = load_data("../data/cifar10/data_batch_2")

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

