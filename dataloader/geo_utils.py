import numpy as np

def RBF(centers, gamma, x):
    centers = centers.reshape(1, -1)
    x = x.reshape(-1, 1)
    diff = x - centers
    return np.exp(-gamma * np.square(diff))
