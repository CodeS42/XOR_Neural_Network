import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

def linear_activation(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def loss_function(predicted, expected):
    return 0.5 * np.square(predicted - expected)

def compute_output_delta(error, z):
    return error * derivative_sigmoid(z)

def backpropagate_error_to_hidden(delta, w, z):
    return np.dot(delta, w.T) * derivative_sigmoid(z)

def update_weights(learning_rate, delta, a):
    return learning_rate * np.outer(delta, a)