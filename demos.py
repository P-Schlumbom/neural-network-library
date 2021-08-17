"""
A collection of functions demonstrating how every class can be used.
"""

import numpy as np
from nn.SOM import SOM

def som_demo():
    """
    Demonstration of the SOM class. Loads a dataset of animal attributes and clusters species by similarity along a
    single axis. This is then printed as a list; note that similar species (e.g. birds, mammals, insects) occur near
    each other in the list.
    Dataset in the demonstration is the Zoo Data Set from the UCI Machine Learning Repository: 
    https://archive.ics.uci.edu/ml/datasets/zoo
    :return:
    """
    datapath = "data/animals/animals.dat"
    animalnamespath = "data/animals/animalnames.txt"

    with open(datapath, 'r') as f:
        data = f.read().split(',')

    with open(animalnamespath, 'r') as f:
        animalnames = f.readlines()
    for i in range(len(animalnames)):
        animalnames[i] = animalnames[i].strip()[1:-1]

    data = np.asarray(data)
    data = data.reshape((32, 84))
    data = data.astype('float64')

    test_som = SOM(84, (100,), init_neighbourhood_size=50)

    test_som.train(data, epochs=20)

    locations = test_som.get_matches(data)
    print(locations)
    name2index = {}
    index2name = {}
    for i in range(len(locations)):
        name2index[animalnames[i]] = locations[i]
        if locations[i] in index2name:
            index2name[locations[i]].append(animalnames[i])
        else:
            index2name[locations[i]] = [animalnames[i]]

    locations.sort()
    for loc in locations:
        print("{}: {}".format(loc, index2name[loc]))

if __name__=="__main__":
    som_demo()

