import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def linear_activation(inputs, weights, bias):
    return np.dot(inputs, weights) + bias
