import numpy

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def derivative_sigmoid(z):
    sigmoid = sigmoid(z)
    return sigmoid * (1 - sigmoid)