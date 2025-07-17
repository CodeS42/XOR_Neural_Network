import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def linear_activation(w, b, input1, input2):
    z1 = input1 * w[0, 0] + input2 * w[1, 0] + b[0]
    z2 = input1 * w[0, 1] + input2 * w[1, 1] + b[1]
    z = [z1, z2]

    return z

def linear_activation_output(w, b, input1, input2):
    return input1 * w[0] + input2 * w[1] + b
