"""
Self Organising Map (SOM)

"""

import numpy as np

VERBOSE = 1

class SOM:
    def __init__(self, input_size, map_shape, eta=0.2,
                 init_neighbourhood_size=None, min_neighbourhood_size=1, decay_rate=0.8,
                 wraparound=False):
        """
        Initialise the Self Organising Map (SOM) with the appropriate parameters.
        :param input_size: int, Size of the input, i.e. the number of input features
        :param map_shape: tuple, shape of the output map. Can be 1D or 2D.
        :param eta: float, learning rate
        :param init_neighbourhood_size: int, initial neighbourhood radius. If left as 'None', all nodes will
        initially be considered part of the neighbourhood.
        :param min_neighbourhood_size: int, the minimum neighbourhood size, which the neighbourhood size will
        eventually shrink to.
        :param decay_rate: float, the rate at which the neighbourhood size is reduced with each epoch.
        :param wraparound: bool, whether or not distances in the output map wrap around
        """

        self.map_shape = map_shape
        self.map_size = np.prod(map_shape)
        self.map_dims = len(map_shape)

        self.W = np.random.normal(0, 1, (self.map_size, input_size))  # SOM weights
        #self.W = np.random.choice([-1., 1.], (self.map_size, input_size))
        self.activations = np.zeros((self.map_size, 1))
        self.distances = self.activations  # note that if input is in range (-1,1) then activations are dot products
        # and largest activation represents most similar weight set, in which case this is redundant
        self.node_distances = np.zeros((self.map_size, self.map_size))
        self.wraparound = wraparound
        self.compute_node_distances()

        self.init_neighbourhood_size = np.max(self.node_distances) if init_neighbourhood_size is None \
            else init_neighbourhood_size  # how many neighbours to consider for update
        self.min_neighbourhood_size = min_neighbourhood_size
        self.neighbourhood_size = self.init_neighbourhood_size
        self.eta = eta
        self.decay_rate = decay_rate

    def compute_node_distances(self):
        """
        For each node, compute the distance to every other node in the output array and store the result.
        Currently computes for 1D and 2D case.
        :return:
        """

        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map_dims == 1:
                    self.node_distances[i, j] = np.abs(i - j)
                    if self.wraparound and np.abs(i - j) > self.map_size // 2:
                        #self.node_distances[i, j] = np.abs(i - j) % (self.map_size // 2)
                        self.node_distances[i, j] = self.map_size - np.abs(i - j)  # in case of wraparound, nodes
                        # more than half max distance away can actually be reached faster by going the other way
                elif self.map_dims == 2:
                    if i == j:
                        self.node_distances[i, j] = 0
                    else:
                        i_loc_x = i % self.map_shape[1]  # x-location in output grid
                        i_loc_y = i // self.map_shape[1]  # y-location in output grid
                        j_loc_x = j % self.map_shape[1]  # x-location of j-node in output grid
                        j_loc_y = j // self.map_shape[1]  # y-location of j-node in output grid
                        x_dist = np.abs(i_loc_x - j_loc_x)
                        if self.wraparound and x_dist > self.map_shape[0] // 2:
                            x_dist = self.map_shape[0] - x_dist
                        y_dist = np.abs(i_loc_y - j_loc_y)
                        if self.wraparound and y_dist > self.map_shape[1] // 2:
                            y_dist = self.map_shape[1] - y_dist
                        h_dist = x_dist + y_dist
                        self.node_distances[i, j] = h_dist
                else:
                    raise Exception("Only up to 2D output maps supported!")

    def forward(self, X):
        """
        Compute the output node activations for a given input
        :param X: numpy array of shape (#_features,1) (Can in principle be of shape (#_features, #_inputs)
        :return: numpy array of shape (#_inputs), so ordinarily just 1
        """
        self.activations = np.matmul(self.W, X)
        return self.activations

    def get_best_match(self, X):
        """
        Find the index of the winning node with the closest match to the input pattern
        :param X: numpy array of shape (#_features,1)
        :return: index of the winning output node
        """
        self.forward(X)
        diff = self.W - X.transpose()
        d = np.matmul(diff, diff.transpose()).diagonal()  # squared distance of each node
        if VERBOSE >= 2: print("min dist: {:.3g}".format(np.min(d)))

        winner = np.argmin(d)

        return winner, d[winner]

    def get_neighbours(self, winner):
        """
        Find the indices of all output nodes in the winning node's neighbourhood (including the winning node)
        :param winner: the index of the winning node
        :return: numpy array of all neighbourhood indices
        """
        neighbours = np.where(self.node_distances[winner] <= self.neighbourhood_size)
        return neighbours

    def update_weights(self, X, neighbours):
        """
        Update the weights of all the neurons in the winning neighbourhood (including the winner)
        :param X: numpy array of shape (#_features,)
        :param neighbours: numpy array of indices of the winner's neighbours (includes the winner's index as well)
        :return:
        """
        update = self.eta * (X.transpose() - self.W)
        mask = np.zeros(self.W.shape)
        mask[neighbours] = 1
        update *= mask
        self.W += update

    def sample_train(self, X):
        """
        Carry out the training procedure for a single sample. SOMs should process data sample by sample.
        :param X: numpy array of shape (#_inputs,)
        :return:
        """
        X = X.reshape((-1,1))  # add necessary dimension
        winner, _ = self.get_best_match(X)  # select the winning output neuron
        neighbours = self.get_neighbours(winner)  # select all neurons in the neighbourhood of the winning neuron
        self.update_weights(X, neighbours)  # update the weights for all the selected neurons

    def train(self, X, epochs=10):
        """
        Train the SOM on a given dataset
        :param X: numpy array of shape (# inputs, # features)
        :return:
        """
        for epoch in range(epochs):
            mean_dist = 0
            for x in X:
                self.sample_train(x)
                mean_dist += self.get_best_match(x)[1]
            mean_dist = mean_dist / len(X)
            if VERBOSE >= 1: print("epoch {}: mean best distance: {:.3g}".format(epoch, mean_dist))
            self.neighbourhood_size = max(self.neighbourhood_size * self.decay_rate, self.min_neighbourhood_size)  #
            # reduce the neighbourhood size
            #print(self.neighbourhood_size)


    def get_matches(self, X):
        """
        Given a set of inputs, return the index of the winning output node for each input
        :param X: numpy array of shape (# inputs, # features)
        :return: list of length #_inputs, containing the winning node index for each input
        """
        indexes = []
        for x in X:
            indexes.append(self.get_best_match(x)[0])
        return indexes