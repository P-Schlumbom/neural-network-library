"""
Class implementing training regime
"""

from global_params import *  # sets printing status
import numpy as np

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
